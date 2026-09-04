// Minimal persistent thread pool with a parallel-for primitive.
// Workers are numbered 0..n-1 so callers can keep per-worker state (simulators).
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gpw {

class ThreadPool {
public:
    explicit ThreadPool(int n_workers) : n_(n_workers < 1 ? 1 : n_workers) {
        // Worker 0 is the calling thread; workers 1..n-1 are real threads.
        for (int i = 1; i < n_; ++i) {
            threads_.emplace_back([this, i] { WorkerLoop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

    int Size() const { return n_; }

    // Runs fn(task_index, worker_index) for task_index in [0, n_tasks).
    // Blocks until all tasks are done. Not re-entrant.
    void ParallelFor(int n_tasks, const std::function<void(int, int)>& fn) {
        if (n_tasks <= 0) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_tasks_ = n_tasks;
            next_.store(0);
            active_ = n_ - 1;
            ++generation_;
        }
        cv_.notify_all();
        Drain(0);
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return active_ == 0; });
        fn_ = nullptr;
    }

private:
    void Drain(int worker) {
        const std::function<void(int, int)>* fn = fn_;
        int n = n_tasks_;
        while (true) {
            int i = next_.fetch_add(1);
            if (i >= n) break;
            (*fn)(i, worker);
        }
    }

    void WorkerLoop(int worker) {
        unsigned long long seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return generation_ != seen; });
                seen = generation_;
                if (quit_) return;
            }
            Drain(worker);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--active_ == 0) done_cv_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    unsigned long long generation_ = 0;
    bool quit_ = false;
    const std::function<void(int, int)>* fn_ = nullptr;
    int n_tasks_ = 0;
    std::atomic<int> next_{0};
    int active_ = 0;
};

}  // namespace gpw
