#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>

namespace wheels {

    template <class FunT>
    void parallel_for_each(size_t n, FunT && fun, size_t batch_num = 1, 
        size_t concurrency_num = std::thread::hardware_concurrency()) {
        std::vector<std::thread> threads;
        threads.reserve(concurrency_num);
        for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
            size_t bfirst = bid * batch_num;
            size_t blast = std::min(n, (bid + 1) * batch_num) - 1;
            threads.emplace_back([&fun](size_t first, size_t last) {
                for (size_t i = first; i <= last; i++) {
                    fun(i);
                }
            }, bfirst, blast);
            if (threads.size() >= concurrency_num || bid == n / batch_num) {
                for (auto & t : threads) {
                    t.join();
                }
                threads.clear();
            }
        }
    }


    template <class IterT, class FunT>
    void parallel_for_each(IterT begin, IterT end, FunT && fun, size_t batch_num = 1,
        size_t concurrency_num = std::thread::hardware_concurrency()) {        
        std::vector<std::thread> threads;
        threads.reserve(concurrency_num);

        auto n = std::distance(begin, end);
        for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
            size_t bfirst = bid * batch_num;
            size_t blast = std::min(n, (bid + 1) * batch_num) - 1;
            threads.emplace_back([&fun](Iter first, Iter last) {
                while(first != last){
                    fun(*first);
                    ++first;
                }
            }, begin + bfirst, begin + blast);
            if (threads.size() >= concurrency_num || bid == n / batch_num) {
                for (auto & t : threads) {
                    t.join();
                }
                threads.clear();
            }
        }
    }


    template <class IterT, class T, class ReduceT>
    T parallel_accumulate(IterT begin, IterT end, const T & initial, ReduceT && redux, size_t batch_num = 1,
        size_t concurrency_num = std::thread::hardware_concurrency()) {
        std::vector<std::future<T>> futures;
        futures.reserve(concurrency_num);
        
        T result = initial;
        auto n = std::distance(begin, end);
        for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
            size_t bfirst = bid * batch_num;
            size_t blast = std::min<size_t>(n, (bid + 1) * batch_num) - 1;
            std::packaged_task<T(IterT, IterT)> task([&redux](IterT first, IterT last) {
                T tmp_result = 0;
                while(first != last) {
                    tmp_result = redux(tmp_result, *first);
                    ++first;
                }
                return tmp_result;
            });

            futures.push_back(task.get_future());
            
            std::thread(std::move(task), begin + bfirst, begin + blast).detach();
            if (futures.size() >= concurrency_num || bid == n / batch_num) {
                for (auto & f : futures) {
                    f.wait();
                }
                for (auto & f : futures) {
                    result = redux(result, f.get());
                }
                futures.clear();
            }
        }
        return result;
    }


}