"""
Async Memory Queue - Non-blocking memory operations for real-time voice pipeline.

Memory writes (embedding generation + DB writes) take ~5-6 seconds.
This queue allows the pipeline to continue while memory operations happen in background.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any
from datetime import datetime


@dataclass
class MemoryTask:
    """A memory operation task."""
    task_id: str
    operation: Callable[[], Any]
    priority: int = 1  # 1=high, 2=normal, 3=low
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def __lt__(self, other):
        # Higher priority = lower number (processed first)
        return self.priority < other.priority


class AsyncMemoryQueue:
    """
    Non-blocking queue for memory operations.

    Usage:
        queue = AsyncMemoryQueue()
        queue.enqueue(lambda: memory_engine.ingest_event(...))  # Non-blocking!
        # Continue pipeline immediately
    """

    def __init__(self, num_workers: int = 2):
        """
        Initialize async memory queue.

        Args:
            num_workers: Number of background worker threads (default 2)
        """
        self.queue = queue.PriorityQueue()
        self.num_workers = num_workers
        self.workers = []
        self.running = True
        self.task_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()

        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncMemory-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self):
        """Worker thread that processes memory tasks."""
        while self.running:
            try:
                # Get task with timeout to allow graceful shutdown
                task = self.queue.get(timeout=1.0)

                if task is None:  # Shutdown signal
                    break

                # Execute memory operation
                try:
                    task.operation()
                    with self.lock:
                        self.completed_count += 1
                except Exception as e:
                    print(f"[AsyncMemoryQueue] Task {task.task_id} failed: {e}")
                    with self.lock:
                        self.failed_count += 1
                finally:
                    self.queue.task_done()

            except queue.Empty:
                continue  # No tasks, keep waiting

    def enqueue(
        self,
        operation: Callable[[], Any],
        priority: int = 1,
        task_id: Optional[str] = None
    ) -> str:
        """
        Enqueue a memory operation (non-blocking).

        Args:
            operation: Function to execute (e.g., lambda: memory_engine.ingest_event(...))
            priority: 1=high, 2=normal, 3=low
            task_id: Optional task identifier

        Returns:
            task_id for tracking
        """
        if task_id is None:
            with self.lock:
                self.task_count += 1
                task_id = f"memory_task_{self.task_count}"

        task = MemoryTask(
            task_id=task_id,
            operation=operation,
            priority=priority
        )

        self.queue.put(task)
        return task_id

    def enqueue_memory_write(
        self,
        memory_engine,
        user_id: int,
        role: str,
        text: str,
        session_id: str,
        ts: datetime,
        temporal_tags: dict,
        source: str = "async",
        metadata: dict = None,
        priority: int = 1
    ) -> str:
        """
        Convenience method to enqueue memory ingestion.

        Returns:
            task_id for tracking
        """
        def _write():
            return memory_engine.ingest_event(
                user_id=user_id,
                role=role,
                text=text,
                session_id=session_id,
                ts=ts,
                temporal_tags=temporal_tags,
                source=source,
                metadata=metadata or {}
            )

        return self.enqueue(_write, priority=priority)

    def enqueue_adaptation_update(
        self,
        adaptation_engine,
        user_id: int,
        metrics,
        priority: int = 2
    ) -> str:
        """
        Convenience method to enqueue adaptation updates.

        Returns:
            task_id for tracking
        """
        def _update():
            return adaptation_engine.update(user_id, metrics)

        return self.enqueue(_update, priority=priority)

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all queued tasks to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        try:
            if timeout is None:
                # No timeout - wait forever
                self.queue.join()
                return True
            else:
                # Wait with timeout
                start = time.time()
                while not self.queue.empty():
                    if time.time() - start > timeout:
                        return False
                    time.sleep(0.1)
                # Final join with remaining time
                remaining = timeout - (time.time() - start)
                if remaining > 0:
                    self.queue.join()
                return self.queue.empty()
        except Exception as e:
            print(f"[AsyncMemoryQueue] wait_for_completion error: {e}")
            return False

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self.lock:
            return {
                "queued": self.queue.qsize(),
                "total_tasks": self.task_count,
                "completed": self.completed_count,
                "failed": self.failed_count,
                "workers": self.num_workers
            }

    def shutdown(self, wait: bool = True):
        """Shutdown the queue and workers."""
        self.running = False

        # Send shutdown signal to all workers
        for _ in self.workers:
            try:
                # Use high priority for shutdown signal to bypass normal tasks
                shutdown_task = MemoryTask(
                    task_id="shutdown",
                    operation=lambda: None,
                    priority=0  # Highest priority
                )
                self.queue.put(shutdown_task)
            except:
                pass

        if wait:
            for worker in self.workers:
                worker.join(timeout=5.0)


# Global singleton instance
_global_async_memory_queue: Optional[AsyncMemoryQueue] = None


def get_async_memory_queue(num_workers: int = 2) -> AsyncMemoryQueue:
    """Get or create global async memory queue."""
    global _global_async_memory_queue
    if _global_async_memory_queue is None:
        _global_async_memory_queue = AsyncMemoryQueue(num_workers=num_workers)
    return _global_async_memory_queue
