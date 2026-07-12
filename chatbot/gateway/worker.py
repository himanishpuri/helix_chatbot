# gateway/worker.py
import os
import asyncio
import json
import httpx
import redis.asyncio as aioredis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
QUEUE_KEY = "inference_queue"
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/completion")
NUM_WORKERS = 8
CANCEL_CHECK_EVERY = 8  # tokens between cancel-flag checks

redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


async def process_job(job: dict):
    """Call llama-server, stream tokens to the Redis pub/sub channel.

    Always publishes a terminal `[DONE]` (success, error, or cancel) so the
    gateway's stream generator can't hang waiting for a sentinel llama.cpp's
    /completion endpoint never sends.
    """
    session_id = job["session_id"]
    prompt = job["prompt"]
    channel = f"response:{session_id}"
    cancel_key = f"cancel:{session_id}"

    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "stream": True,
        "temperature": 0.2,
        "stop": ["<|end|>", "<|user|>"],
    }

    try:
        if await redis_client.exists(cancel_key):
            return  # client already gone; skip inference entirely

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", LLAMA_URL, json=payload) as response:
                seen = 0
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":  # OpenAI-compat servers
                        return
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("content", "")
                    if token:
                        await redis_client.publish(
                            channel, json.dumps({"token": token})
                        )
                        seen += 1
                        if seen % CANCEL_CHECK_EVERY == 0 and await redis_client.exists(
                            cancel_key
                        ):
                            print(f"[worker] cancelled session={session_id}")
                            return  # abort: exiting closes the llama.cpp stream
                    if data.get("stop") is True:  # llama.cpp /completion terminator
                        return
    except Exception as e:
        await redis_client.publish(channel, json.dumps({"error": str(e)}))
    finally:
        # End of stream / stop / cancel / error all converge here.
        await redis_client.publish(channel, "[DONE]")


async def worker(worker_id: int):
    print(f"[worker {worker_id}] started, waiting for jobs...")
    while True:
        try:
            # Blocking pop — waits until a job appears
            result = await redis_client.brpop(QUEUE_KEY, timeout=5)
            if result is None:
                continue
            _, raw_job = result
            job = json.loads(raw_job)
            print(f"[worker {worker_id}] processing session={job['session_id']}")
            await process_job(job)
        except Exception as e:
            print(f"[worker {worker_id}] error: {e}")
            await asyncio.sleep(1)


async def main():
    tasks = [asyncio.create_task(worker(i)) for i in range(NUM_WORKERS)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
