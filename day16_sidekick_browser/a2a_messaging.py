"""
A2A Multi-Agent Communication Layer
Supports: RabbitMQ (AMQP) and Redis Streams

Agents publish/subscribe through a shared message broker.
This module is the backbone for the D71-75 multi-agent analytics system.

Usage:
  # Publisher (BrowserAgent sends scraped data)
  broker = get_broker("redis")          # or "rabbitmq"
  await broker.publish("data.scraped", {"url": "...", "text": "..."})

  # Subscriber (AnalystAgent listens)
  async for msg in broker.subscribe("data.scraped"):
      stats = analyse(msg)
      await broker.publish("data.analysed", stats)
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


# ─── Message Envelope ─────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    topic:     str
    sender:    str
    payload:   Dict[str, Any]
    msg_id:    str = field(default_factory=lambda: f"msg-{int(time.time()*1000)}")
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str) -> "AgentMessage":
        return AgentMessage(**json.loads(raw))


# ─── Abstract Broker ─────────────────────────────────────────────────────────

class AgentBroker(ABC):
    @abstractmethod
    async def publish(self, topic: str, payload: Dict, sender: str = "agent") -> None: ...

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[AgentMessage]: ...

    @abstractmethod
    async def close(self) -> None: ...


# ─── In-Process Broker (no external service needed for demo) ─────────────────

class InProcessBroker(AgentBroker):
    """
    Pure-Python in-memory broker — perfect for unit tests and demos.
    Swap for RedisBroker or RabbitMQBroker in production.
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def _queue(self, topic: str) -> asyncio.Queue:
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        return self._queues[topic]

    async def publish(self, topic: str, payload: Dict, sender: str = "agent") -> None:
        msg = AgentMessage(topic=topic, sender=sender, payload=payload)
        await self._queue(topic).put(msg)
        print(f"  [Broker- >{topic}] {sender} published {list(payload.keys())}")

    async def subscribe(self, topic: str) -> AsyncIterator[AgentMessage]:
        q = self._queue(topic)
        while True:
            msg = await asyncio.wait_for(q.get(), timeout=10)
            yield msg

    async def close(self) -> None:
        self._queues.clear()


# ─── Redis Streams Broker ─────────────────────────────────────────────────────

class RedisBroker(AgentBroker):
    """
    Redis Streams-backed broker.
    Requires: pip install redis[asyncio]
    Set REDIS_URL env var (default: redis://localhost:6379)
    """

    def __init__(self, url: str = "redis://localhost:6379"):
        self.url = url
        self._redis: Optional[Any] = None

    async def _connect(self):
        if self._redis is None:
            import redis.asyncio as aioredis          # lazy import
            self._redis = aioredis.from_url(self.url, decode_responses=True)

    async def publish(self, topic: str, payload: Dict, sender: str = "agent") -> None:
        await self._connect()
        msg = AgentMessage(topic=topic, sender=sender, payload=payload)
        # Redis Streams XADD — fields must be flat strings
        await self._redis.xadd(f"stream:{topic}", {"data": msg.to_json()})
        print(f"  [Redis- >{topic}] {sender} xadd ok")

    async def subscribe(self, topic: str) -> AsyncIterator[AgentMessage]:
        await self._connect()
        stream = f"stream:{topic}"
        last_id = "0"
        while True:
            entries = await self._redis.xread({stream: last_id}, block=5_000, count=10)
            for _, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id
                    yield AgentMessage.from_json(fields["data"])

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()


# ─── RabbitMQ Broker ─────────────────────────────────────────────────────────

class RabbitMQBroker(AgentBroker):
    """
    RabbitMQ AMQP-backed broker using topic exchanges.
    Requires: pip install aio-pika
    Set RABBITMQ_URL env var (default: amqp://guest:guest@localhost/)
    """

    def __init__(self, url: str = "amqp://guest:guest@localhost/"):
        self.url = url
        self._conn = None
        self._channel = None

    async def _connect(self):
        if self._conn is None:
            import aio_pika                           # lazy import
            self._conn    = await aio_pika.connect_robust(self.url)
            self._channel = await self._conn.channel()
            self._exchange = await self._channel.declare_exchange(
                "sidekick", aio_pika.ExchangeType.TOPIC, durable=True
            )

    async def publish(self, topic: str, payload: Dict, sender: str = "agent") -> None:
        await self._connect()
        import aio_pika
        msg = AgentMessage(topic=topic, sender=sender, payload=payload)
        await self._exchange.publish(
            aio_pika.Message(body=msg.to_json().encode()),
            routing_key=topic,
        )
        print(f"  [RabbitMQ- >{topic}] {sender} published")

    async def subscribe(self, topic: str) -> AsyncIterator[AgentMessage]:
        await self._connect()
        import aio_pika
        queue = await self._channel.declare_queue("", exclusive=True)
        await queue.bind(self._exchange, routing_key=topic)
        async with queue.iterator() as q_iter:
            async for raw_msg in q_iter:
                async with raw_msg.process():
                    yield AgentMessage.from_json(raw_msg.body.decode())

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_broker(backend: str = "inprocess", **kwargs) -> AgentBroker:
    backends = {
        "inprocess": InProcessBroker,
        "redis":     RedisBroker,
        "rabbitmq":  RabbitMQBroker,
    }
    cls = backends.get(backend.lower())
    if cls is None:
        raise ValueError(f"Unknown broker '{backend}'. Choose: {list(backends)}")
    return cls(**kwargs)


# ─── Demo Pipeline (self-contained) ──────────────────────────────────────────

async def demo():
    """
    3-agent pipeline over InProcessBroker:
      BrowserAgent - > data.scraped - > AnalystAgent - > data.analysed - > ReporterAgent
    """
    print("\n" + "="*60)
    print(" A2A Multi-Agent Demo  (InProcess broker)")
    print("="*60)

    broker = get_broker("inprocess")

    async def browser_agent():
        """Simulates scraping, then publishes result."""
        await asyncio.sleep(0.1)
        await broker.publish("data.scraped", {
            "url": "https://example.com/ai-report",
            "title": "AI Trends 2025",
            "word_count": 1_420,
            "snippet": "Artificial intelligence adoption surged 38 % in 2024…",
        }, sender="BrowserAgent")

    async def analyst_agent():
        """Consumes scraped data, publishes analytics."""
        async for msg in broker.subscribe("data.scraped"):
            p = msg.payload
            stats = {
                "source": p["url"],
                "words":  p["word_count"],
                "grade":  "RICH" if p["word_count"] > 1_000 else "THIN",
            }
            await broker.publish("data.analysed", stats, sender="AnalystAgent")
            break  # one message demo

    async def reporter_agent():
        """Consumes analytics, prints final report."""
        async for msg in broker.subscribe("data.analysed"):
            print(f"\n[ReporterAgent] Final report:")
            print(json.dumps(msg.payload, indent=2))
            break

    await asyncio.gather(
        browser_agent(),
        analyst_agent(),
        reporter_agent(),
    )
    await broker.close()
    print("\n A2A demo complete")


if __name__ == "__main__":
    asyncio.run(demo())
