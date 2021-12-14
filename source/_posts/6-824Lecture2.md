---
title: 6.824Lecture2-RPC&Threads
mathjax: false
date: 2021-12-11 22:25:55
tags: 6.824
categories: Learning
---

# Background

## Why Threads?
- I/O concurrency
- multi core parallelism
- convenience for, like routine

<!-- more -->

## Threads Changes 
- race conditions
  - avoid share memory
  - use locks: 并行变串行
- coordinations
  - channels or condition variables
- dead lock

# Go Methods 

## Channel
- 用来在并发中同步变量
- pipeline with datatype, need be created, like: ch = make(chan int)
- receivers & sender.默认情况下，发送和接收操作在另一端准备好之前都会阻塞。这使得 Go 程可以在没有显式的锁或竞态变量的情况下进行同步
  - sender: 通过close关闭信道表示没有要发送的值. **向一个已经关闭的信道发送数据会引发程序panic**
  - receiver: v, ok := <-ch, ok=false when channel is empty and is closed

## sync.Mutex
- 互斥锁，代码段前后调用Lock() & Unlock()实现某段代码的互斥执行，用defer保证互斥锁一定被解锁。
  - defer func()：func会在包含defer语句的函数返回时再执行

## sync.Cond
- Wait()/Broadcast()/Singal()

# RPC
*remote procedure call*
- goal: rpc ≈ pc
- pros: stub完成数据转换和解析、打开/关闭连接等细节

## workflow
![workflow](https://pic1.zhimg.com/45366c44f775abfd0ac3b43bccc1abc3_r.jpg?source=1940ef5c)

1. 客户端调用 client stub，并将调用参数 push 到栈（stack）中，这个调用是在本地的
2. client stub 将这些参数包装，并通过系统调用发送到服务端机器。打包的过程叫 marshalling（常见方式：XML、JSON、二进制编码）。
3. 客户端操作系统将消息传给传输层，传输层发送信息至服务端；
4. 服务端的传输层将消息传递给 server stub
5. server stub 解析信息。该过程叫 unmarshalling。
6. server stub 调用程序，并通过类似的方式返回给客户端。
7. 客户端拿到数据解析后，将执行结果返回给调用者。

## RPC Semantics under failures
**server不知道client的具体情况，导致failure时难以处理**
- at-least-once: client不断重试直到ack成功。适用于幂等操作
  - 幂等操作：幂等操作的特点是其任意多次执行所产生的影响均与一次执行的影响相同
- at-most-once: **最常见的**，只执行0/1次。通过过滤重复项来实现。
- exactly-once: hard

go rpc: at-most-once, 由client手动决定是否重试 

# Refs
- [note](https://mp.weixin.qq.com/s?__biz=MzIwODA2NjIxOA==&mid=2247484193&idx=1&sn=693e0ff4bfcc6e02dea10ed9d639b41b&chksm=970980e4a07e09f2647de63ed0bf3be98d9032a3797033af3872c692d2373f98627a63f30e22&scene=178&cur_album_id=1751707148520112128#rd)
- [video](https://www.bilibili.com/video/BV1e5411E7RM?p=2&spm_id_from=pageDriver)
