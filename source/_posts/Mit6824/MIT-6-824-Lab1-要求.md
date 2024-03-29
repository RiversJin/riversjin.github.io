---
title: MIT 6.824 Lab1 要求
date: 2021-10-11 20:35:19
categories: 
- "MIT 6.824"
tags:
- "分布式"
---
这一篇文章主要是对实验1要求的翻译 以及顺便需要记录的碎碎念.

## Introduction
> In this lab you'll build a MapReduce system. You'll implement a worker process that calls application Map and Reduce functions and handles reading and writing files, and a coordinator process that hands out tasks to workers and copes with failed workers. You'll be building something similar to the MapReduce paper. (Note: the lab uses "coordinator" instead of the paper's "master".)

在这个实验中,你需要实现一个MapReduce系统.你需要实现一个调用*Map*和*Reduce*的worker process. 以及一个负责将任务分发到worker process并处理work process错误的coordinator process. 

## Getting started
> You'll fetch the initial lab software with git (a version control system). To learn more about git, look at the Pro Git book or the git user's manual. To fetch the 6.824 lab software:

通过git拉取代码,如下:
```bash
$ git clone git://g.csail.mit.edu/6.824-golabs-2021 6.824
$ cd 6.824
$ ls
Makefile src
$
```
>We supply you with a simple sequential mapreduce implementation in src/main/mrsequential.go. It runs the maps and reduces one at a time, in a single process. We also provide you with a couple of MapReduce applications: word-count in mrapps/wc.go, and a text indexer in mrapps/indexer.go. You can run word count sequentially as follows:

在src/main/mrsequential.go中,以及实现了一个简单的顺序的MapReduce实现(即非并发的版本).它在一个进程中,执行*map*和*reduce*.

另外,还提供了一些MapReduce的应用程序,比如mrapps/wc.go的word count,以及mrapps/indexer.go的文本索引器.

以下的命令可以以顺序版本执行word count:
```bash
$ cd ~/6.824
$ cd src/main
$ go build -race -buildmode=plugin ../mrapps/wc.go
$ rm mr-out*
$ go run -race mrsequential.go wc.so pg*.txt
$ more mr-out-0
A 509
ABOUT 2
ACT 8
...
```
**注意: 如果编译的时候没有-race参数,那么运行的时候也不可加-race参数**
> -race参数是golang内置的竞态检测,一般来说,内存使用增加5-10倍,运行速度减慢2-20倍.  
具体参考 https://golang.org/doc/articles/race_detector

## Your Job
> Your job is to implement a distributed MapReduce, consisting of two programs, the coordinator and the worker. There will be just one coordinator process, and one or more worker processes executing in parallel. In a real system the workers would run on a bunch of different machines, but for this lab you'll run them all on a single machine. The workers will talk to the coordinator via RPC. Each worker process will ask the coordinator for a task, read the task's input from one or more files, execute the task, and write the task's output to one or more files. The coordinator should notice if a worker hasn't completed its task in a reasonable amount of time (for this lab, use ten seconds), and give the same task to a different worker.

你的工作是实现一个分布式的MapReduce.它主要由Worker和Coordinator组成. 一个Coordinator与若干个Worker并行执行. 在真实系统中,这些进程会在不同的机器上运行,但是对于这个实验来说,Coordinator与Worker会在同一个机器上执行. 进程间通过RPC通信.Workder会向Coordinator请求发布任务,从一个或若干个文件中读取输入,执行任务,并将输出写另一些文件中.Coordinator要注意到每个Workder是否在合理的时间内完成任务(在这个实验中,这个时间是10s),如果没有,将这个任务交给另一个Worker

>We have given you a little code to start you off. The "main" routines for the coordinator and worker are in main/mrcoordinator.go and main/mrworker.go; don't change these files. You should put your implementation in mr/coordinator.go, mr/worker.go, and mr/rpc.go.

我们已经给了你一小部分代码.Worker和Coordinator的main routine位于main/mrcoordinator.go和main/mrworker.go中.不要更改这些文件。您应该将实现放在mr/coordinator.go、mr/worker.go和mr/rpc.go中。

> Here's how to run your code on the word-count MapReduce application. First, make sure the word-count plugin is freshly built:

以下,介绍如何使用你的代码运行word count.首先,确保word-count是最新构建出的:
```bash
$ go build -race -buildmode=plugin ../mrapps/wc.go
```

> In the main directory, run the coordinator.
在main目录中,运行Coordinator程序
``` bash
$ rm mr-out*
$ go run -race mrcoordinator.go pg-*.txt
```

> The pg-*.txt arguments to mrcoordinator.go are the input files; each file corresponds to one "split", and is the input to one Map task. The -race flags runs go with its race detector.

传入"pg-*.txt"参数到mrcoordinator.go是输入文件,每个文件都是一个"切片"(即输入文件已经被拆分为若干个小文件了),这对应一个Map task. -race 参数告诉编译器启用竞态检测.

> In one or more other windows, run some workers:

在其他窗口上,执行worker进程:
```
$ go run -race mrworker.go wc.so
```

> When the workers and coordinator have finished, look at the output in mr-out-*. When you've completed the lab, the sorted union of the output files should match the sequential output, like this:

当执行成功后,查看mr-out-*中的输出,这些输出结果应该与提供的顺序版本的MapReduce的输出一致.像这样:
```bash
$ cat mr-out-* | sort | more
A 509
ABOUT 2
ACT 8
...
```
>We supply you with a test script in main/test-mr.sh. The tests check that the wc and indexer MapReduce applications produce the correct output when given the pg-xxx.txt files as input. The tests also check that your implementation runs the Map and Reduce tasks in parallel, and that your implementation recovers from workers that crash while running tasks.

我们提供了一个测试脚本,位于main/test-mr.sh.它会检查wc和indexer能否在以pg-xxx.txt文件作为输入时,产生正确的输出. 也会检查你的实现是否是并行的,以及能否在某些worker崩溃后恢复.

>If you run the test script now, it will hang because the coordinator never finishes:

如果现在运行测试脚本，它将挂起，因为还没有完成Coordinator.

> You can change ret := false to true in the Done function in mr/coordinator.go so that the coordinator exits immediately. Then:

你可以将mr/coordinator.go中的Done函数中的reg:=false改为reg:=true.这样Coordinator会立即退出.这样脚本就不会卡住了.

```bash
$ bash test-mr.sh
*** Starting wc test.
sort: No such file or directory
cmp: EOF on mr-wc-all
--- wc output is not the same as mr-correct-wc.txt
--- wc test: FAIL
$
```

> The test script expects to see output in files named mr-out-X, one for each reduce task. The empty implementations of mr/coordinator.go and mr/worker.go don't produce those files (or do much of anything else), so the test fails.

测试脚本期望的输出文件为mr-out-X,每个reduce任务对应一个文件.mr/coordinator.go与mr/worker.go的空实现不会生成这些输出,所以上面的测试会失败.

> When you've finished, the test script output should look like this:

当你完成这个实验后,运行测试脚本,会得到类似于下面的输出:

```bash
$ bash test-mr.sh
*** Starting wc test.
--- wc test: PASS
*** Starting indexer test.
--- indexer test: PASS
*** Starting map parallelism test.
--- map parallelism test: PASS
*** Starting reduce parallelism test.
--- reduce parallelism test: PASS
*** Starting crash test.
--- crash test: PASS
*** PASSED ALL TESTS
$
```
> You'll also see some errors from the Go RPC package that look like

你会在Go PRC包中看到类似于下面的报错:
```bash
2019/12/16 13:27:09 rpc.Register: method "Done" has 1 input parameters; needs exactly three
```
> Ignore these messages; registering the coordinator as an RPC server checks if all its methods are suitable for RPCs (have 3 inputs); we know that Done is not called via RPC.

忽略这些报错即可.将Coordinator注册为RPC server时检查Coordinator中的所有导出方法是否适合RPC的调用,但是我们知道Done并不是通过RPC调用的,所以不必理会.

## A few rules
* The map phase should divide the intermediate keys into buckets for nReduce reduce tasks, where nReduce is the argument that main/mrcoordinator.go passes to MakeCoordinator().

* The worker implementation should put the output of the X'th reduce task in the file mr-out-X.

* worker实现应该将第X个reduce任务的输出放在mr-out-X文件中

* A mr-out-X file should contain one line per Reduce function output. The line should be generated with the Go "%v %v" format, called with the key and value. Have a look in main/mrsequential.go for the line commented "this is the correct format". The test script will fail if your implementation deviates too much from this format.

* You can modify mr/worker.go, mr/coordinator.go, and mr/rpc.go. You can temporarily modify other files for testing, but make sure your code works with the original versions; we'll test with the original versions.

* 你可以修改mr/worker.go、mr/coordinator.go和mr/rpc.go。你可以临时修改其他文件进行测试，但请确保代码与原始版本兼容；我们将使用原始版本进行测试。

* The worker should put intermediate Map output in files in the current directory, where your worker can later read them as input to Reduce tasks.

* Worker应该将中间的Map输出放在当前目录中,这样稍后的Workder可以将其作为Reduce任务的输入

* main/mrcoordinator.go expects mr/coordinator.go to implement a Done() method that returns true when the MapReduce job is completely finished; at that point,mrcoordinator.go will exit.

* main/mrcoordinator.go期望mr/coordinator.go实现一个Done方法.当MapReduce任务完成时,此函数返回true. 整个coordinator.go程序退出.

* When the job is completely finished, the worker processes should exit. A simple way to implement this is to use the return value from call(): if the worker fails to contact the coordinator, it can assume that the coordinator has exited because the job is done, and so the worker can terminate too. Depending on your design, you might also find it helpful to have a "please exit" pseudo-task that the coordinator can give to workers.

* 当任务全部完成时,Worker进程应该退出.实现这一点可以使用call()的返回值: 如果Workder未能联系到Coordinator,可以假设Coordinator已经退出,因为任务已经完成,那么Worker也可以退出.这取决与你的设计,你可能会发现,如果Coordinator可以向Worker发送一个"退出"的伪任务,会很有用.

## Hints
* One way to get started is to modify mr/worker.go's Worker() to send an RPC to the coordinator asking for a task. Then modify the coordinator to respond with the file name of an as-yet-unstarted map task. Then modify the worker to read that file and call the application Map function, as in mrsequential.go.

* 你可以从这个方面入手: 修改mr/worker.go的worker(),通过RPC向Coordinator请求任务,然后修改Coordiantor,向Worker返回尚未启动的map任务对应的文件名.再然后,修改Worker,读取此文件,调用map程序.

* The application Map and Reduce functions are loaded at run-time using the Go plugin package, from files whose names end in .so.

* Map和Reduce应用程序在运行的时候从.so文件中通过Go plugin package动态加载

* If you change anything in the mr/ directory, you will probably have to re-build any MapReduce plugins you use, with something like go build -race -buildmode=plugin ../mrapps/wc.go

* 如果你修改了mr/目录下的内容,可能需要重新构建MapReduce插件,使用类似于go build-race-buildmode=plugin../mrapps/wc.go的命令

* This lab relies on the workers sharing a file system. That's straightforward when all workers run on the same machine, but would require a global filesystem like GFS if the workers ran on different machines.

* 这个实验需要worker程序们共享同一个文件系统.如果worker在同一台机器上,实现这一点很简单.如果在不同的机器上,就需要类似于GFS这种分布式文件系统了.

* A reasonable naming convention for intermediate files is mr-X-Y, where X is the Map task number, and Y is the reduce task number.

* 一个比较合理的中间文件命名规则是这样的: mr-X-Y. 其中X是Map任务号,Y是Reduce任务号.

* The worker's map task code will need a way to store intermediate key/value pairs in files in a way that can be correctly read back during reduce tasks. One possibility is to use Go's encoding/json package. To write key/value pairs to a JSON file:

* Worker的map任务需要一种在文件中存取中间键\值对的方法,你可以使用Go的encoding/json包

* The map part of your worker can use the ihash(key) function (in worker.go) to pick the reduce task for a given key.

* worker的map部分可以使用ihash(key)函数(在worker.go中)为给定的键选择reduce任务

* You can steal some code from mrsequential.go for reading Map input files, for sorting intermedate key/value pairs between the Map and Reduce, and for storing Reduce output in files.

* 你可以借鉴一些mrsequential.go中代码来实现读取map输入,对map,reduce中间的键值对排序,以及将reduce的输入写入到文件(读书人的事情,怎么能叫偷呢)

* The coordinator, as an RPC server, will be concurrent; don't forget to lock shared data.

* 协调器作为RPC服务器会并发执行,别忘了加锁

* Use Go's race detector, with go build -race and go run -race. test-mr.sh by default runs the tests with the race detector.

* 在go build以及go run时加入-race来启用go的竞态检测,提前发现问题.不然test-mr.sh也会检测竞态问题的.

* Workers will sometimes need to wait, e.g. reduces can't start until the last map has finished. One possibility is for workers to periodically ask the coordinator for work, sleeping with time.Sleep() between each request. Another possibility is for the relevant RPC handler in the coordinator to have a loop that waits, either with time.Sleep() or sync.Cond. Go runs the handler for each RPC in its own thread, so the fact that one handler is waiting won't prevent the coordinator from processing other RPCs.

* Worker有时需要等待.比如,reduce任务需要所有map任务执行完成之后才能开始. 可以在每个请求间使用time.sleep(这个Sleep放在Worker或者Coordinator中都行),或者通过条件变量(sync.Cond)进行同步.Go的RPC调用是并发的,所以其中一个调用阻塞时,不会影响处理其他RPC请求.

* The coordinator can't reliably distinguish between crashed workers, workers that are alive but have stalled for some reason, and workers that are executing but too slowly to be useful. The best you can do is have the coordinator wait for some amount of time, and then give up and re-issue the task to a different worker. For this lab, have the coordinator wait for ten seconds; after that the coordinator should assume the worker has died (of course, it might not have).

* Coordinator无法可靠地区分以下几种Worker
    1. 已经崩溃的
    2. 仍然运行但是因为某些原因停止运行
    3. 执行速度太慢而无法使用
    
    所以,最好让Coordinator等待一段时间,如果仍然不能响应,将这个任务重新分配给其他的Worker.在这个实验中,这个时间是10s,在此之后,Coordinator可以认为对应的Worker已经死亡

* If you choose to implement Backup Tasks (Section 3.6), note that we test that your code doesn't schedule extraneous tasks when workers execute tasks without crashing. Backup tasks should only be scheduled after some relatively long period of time (e.g., 10s).

* 如果你选择实现备份任务(3.6节),要注意,当worker执行备份任务并且没有崩溃时,我们会测试你的代码没有调度其他无关的任务.备份任务应该只在一段比较长的时间后被调度(***这段还没看懂,先留个坑,回头再更详细地补充***)

* To test crash recovery, you can use the mrapps/crash.go application plugin. It randomly exits in the Map and Reduce functions.

* 为了测试崩溃恢复,你可以使用mrapps/crash.go崩溃会在map和reduce任务时随机发生

* To ensure that nobody observes partially written files in the presence of crashes, the MapReduce paper mentions the trick of using a temporary file and atomically renaming it once it is completely written. You can use ioutil.TempFile to create a temporary file and os.Rename to atomically rename it.

* 为了确保在程序崩溃时不会出现部分写入的文件,MapReduce论文中提到了使用临时文件并在完全写入后自动重命名的技巧.你可以使用ioutil.TempFile创建一个临时文件,并使用os.Rename对其进行原子地重命名

* test-mr.sh runs all the processes in the sub-directory mr-tmp, so if something goes wrong and you want to look at intermediate or output files, look there. You can modify test-mr.sh to exit after the failing test, so the script does not continue testing (and overwrite the output files).


* test-mr.sh运行子目录mr-tmp中的所有进程,因此如果出现问题,您希望查看中间文件或输出文件,请查看那里.你可以修改test-mr.sh以在测试失败后退出,这样脚本就不会继续测试

* test-mr-many.sh provides a bare-bones script for running test-mr.sh with a timeout (which is how we'll test your code). It takes as an argument the number of times to run the tests. You should not run several test-mr.sh instances in parallel because the coordinator will reuse the same socket, causing conflicts.

* test-mr-many.sh提供了一个用于运行带有超时的test-mr.sh的基本脚本(这就是我们测试代码的方式).它将运行测试的次数作为参数.因为你不应该并行运行多个test-mr.sh实例,因为协调器将重用同一套接字,从而导致冲突.