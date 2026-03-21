# gateway/worker.py
import asyncio
import json
import httpx
import redis.asyncio as aioredis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
QUEUE_KEY  = "inference_queue"
LLAMA_URL  = "http://localhost:8080/completion"
NUM_WORKERS = 8

redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


async def process_job(job: dict):
    """Call llama-server, stream tokens to Redis pub/sub channel."""
    job_id = job.get("job_id")
    if job_id and await redis_client.exists(f"cancelled:{job_id}"):
        print(f"[worker] skipping cancelled job {job_id}")
        return

    session_id = job["session_id"]
    prompt     = job["prompt"]
    channel    = f"response:{session_id}"

    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "stream": True,
        "temperature": 0.2,
        "stop": ["<|end|>", "<|user|>"]
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", LLAMA_URL, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            await redis_client.publish(channel, "[DONE]")
                            return
                        try:
                            data  = json.loads(raw)
                            token = data.get("content", "")
                            if token:
                                await redis_client.publish(channel, json.dumps({"token": token}))
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        await redis_client.publish(channel, json.dumps({"error": str(e)}))
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
