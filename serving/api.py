"""OpenAI-compatible completion/chat endpoints with SSE streaming."""
from __future__ import annotations
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI,HTTPException,Request as HTTPRequest
from fastapi.responses import JSONResponse,StreamingResponse
from pydantic import BaseModel,Field

from .sampling import SamplingParams
from .scheduler import AdmissionError,Request


class CompletionBody(BaseModel):
    model:str='Qwen/Qwen3-0.6B'; prompt:str
    max_tokens:int=Field(16,ge=1,le=4096); temperature:float=Field(0,ge=0)
    top_p:float=Field(1,gt=0,le=1); top_k:int=Field(0,ge=0); repetition_penalty:float=Field(1,gt=0)
    stop_token_ids:list[int]=Field(default_factory=list); seed:int=0; logprobs:int=Field(0,ge=0,le=20)
    stream:bool=False; priority:int=0; timeout_s:float=Field(60,gt=0,le=3600)


class ChatMessage(BaseModel): role:str; content:str
class ChatBody(CompletionBody):
    prompt:str=''; messages:list[ChatMessage]


def create_app(engine,tokenizer,model_id='Qwen/Qwen3-0.6B'):
    @asynccontextmanager
    async def lifespan(_app):
        await engine.start()
        try:
            yield
        finally:
            await engine.close()
    app=FastAPI(title='KernelLab OpenAI-compatible server',lifespan=lifespan)

    @app.get('/health')
    async def health(): return {'status':'ok','model':model_id,'active_requests':engine.active_requests}
    @app.get('/metrics')
    async def metrics(): return engine.snapshot()
    @app.get('/v1/models')
    async def models(): return {'object':'list','data':[{'id':model_id,'object':'model','owned_by':'kernellab'}]}

    async def execute(body,prompt,chat=False):
        if body.model!=model_id: raise HTTPException(404,'model not found')
        ids=tuple(tokenizer.encode(prompt,add_special_tokens=True))
        request_id='cmpl-'+uuid.uuid4().hex
        params=SamplingParams(body.temperature,body.top_k,body.top_p,body.repetition_penalty,
            frozenset(body.stop_token_ids),body.seed,body.logprobs)
        request=Request(request_id,ids,body.max_tokens,params,priority=body.priority,timeout_s=body.timeout_s)
        try: queue=await engine.submit(request)
        except AdmissionError as error: raise HTTPException(503,str(error),headers={'Retry-After':'1'})

        async def events():
            done=False
            try:
                while True:
                    event=await queue.get()
                    if event.finished:
                        done=True
                        payload={'id':request_id,'object':'chat.completion.chunk' if chat else 'text_completion',
                            'choices':[{'index':0,'delta':{} if chat else None,'text':'' if not chat else None,'finish_reason':event.finish_reason}]}
                        yield 'data: '+json.dumps(payload)+'\n\n'; yield 'data: [DONE]\n\n'; return
                    text=tokenizer.decode([event.token_id],skip_special_tokens=False)
                    choice={'index':0,'finish_reason':None,'logprobs':{'token_logprob':event.logprob,'top_logprobs':event.top_logprobs}}
                    if chat: choice['delta']={'content':text}
                    else: choice['text']=text
                    yield 'data: '+json.dumps({'id':request_id,'object':'chat.completion.chunk' if chat else 'text_completion','choices':[choice]})+'\n\n'
            finally:
                if not done: await engine.cancel(request_id)
        if body.stream: return StreamingResponse(events(),media_type='text/event-stream')
        pieces=[]; finish=None
        async for line in events():
            if line=='data: [DONE]\n\n': continue
            payload=json.loads(line[6:]); choice=payload['choices'][0]; finish=choice['finish_reason'] or finish
            pieces.append(choice.get('delta',{}).get('content','') if chat else choice.get('text',''))
        text=''.join(pieces); usage={'prompt_tokens':len(ids),'completion_tokens':len(engine.scheduler.requests[request_id].generated)}
        usage['total_tokens']=usage['prompt_tokens']+usage['completion_tokens']
        choice={'index':0,'finish_reason':finish}
        if chat: choice['message']={'role':'assistant','content':text}
        else: choice['text']=text
        return {'id':request_id,'object':'chat.completion' if chat else 'text_completion','created':int(time.time()),'model':model_id,'choices':[choice],'usage':usage}

    @app.post('/v1/completions')
    async def completions(body:CompletionBody): return await execute(body,body.prompt)
    @app.post('/v1/chat/completions')
    async def chat(body:ChatBody):
        messages=[x.model_dump() for x in body.messages]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) if hasattr(tokenizer,'apply_chat_template') else '\n'.join(f'{x.role}: {x.content}' for x in body.messages)+'\nassistant:'
        return await execute(body,prompt,True)
    @app.delete('/v1/requests/{request_id}')
    async def cancel(request_id):
        try: changed=await engine.cancel(request_id)
        except KeyError: raise HTTPException(404,'request not found')
        return JSONResponse({'id':request_id,'cancelled':changed})
    return app
