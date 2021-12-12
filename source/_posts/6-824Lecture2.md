---
title: 6.824Lecture2 RPC&Threads
mathjax: false
date: 2021-12-11 22:25:55
updated: 2021-12-11 22:25:55
tags: 6.824
categories: Learning
---

# Background

## Why Threads?
- I/O concurrency
- multi core parallelism
- convenience for, like routine

## Threads Changes 
- race conditions
  - avoid share memory
  - use locks: 并行变串行
- coordinations
  - channels or condition variables
- dead lock

# Go Methods 

## sync.Mutex
- 互斥锁，代码段前后调用Lock() & Unlock()实现某段代码的互斥执行，用defer保证互斥锁一定被解锁。
  - defer func()：func会在包含defer语句的函数返回时再执行

## condition variable

