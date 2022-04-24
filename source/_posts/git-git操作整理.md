---
title: git操作整理
date: 2021-12-10
categories: 
    - Tools
tags:  
    - git
---

git操作整理
<!-- more -->

- git log: 记录版本历史
- git reflog：记录操作历史

## 远程库

- git remote add <remote name> xxx 其中<remote name>是git对远程库的命名，origin是默认叫法。这个命名是本地对远程库的一个命名。   
- git remote -v 查看远程库
- git remote rm <remote name>

## 工作区、缓存区

- stage：缓存，add后缓存区和工作区文件一致，commit后缓存区清空，进入下一个版本

### diff

- 工作区 ↔ 缓存区：git diff，当前修改对比上次add都有啥，git add后当前工作区的文件就不能通过git diff查到了
- 缓存区 ↔ 版本库(HEAD)：git diff --cached 当前add的文件和上个版本的diff
- 工作区 ↔ 库(HEAD)：git diff HEAD -- filename
- 库 ↔ 库：git diff 243550a 24bc01b filename     #较旧的id 较新的id

## 文件修改撤销

### 文件修改撤销（老版本git）

#### checkout --（撤销工作区修改）

- 不加--可能变成切换分支
- 文件回到最近一次git commit或git add时的状态。
  - 修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态（git commit）；
  - 已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态（git add）。

#### reset HEAD <fileName>（撤销暂存区修改）

- **把暂存区的文件回退到工作区**
- 再利用git checkout --就可以删除修改了
- 或者直接用git checkout HEAD <fileName> 直接从HEAD恢复文件

### 文件修改撤销（新版本git）

- 撤销工作区的修改：git restore <fileName>
- 撤销工作区和暂存区的修改，恢复到HEAD：git resotre --worktree <fileName>
- 撤销暂存区的修改，前提是上次add后该文件未做其他修改：git restore --staged <fileName>
- 从master同时恢复工作区和暂存区：git restore --source=HEAD --staged --worktree <fileName>
  
## 版本控制

### 回退

- HEAD始终指向当前分支的最新commit，HEAD^ 上一个commit，HEAD~100 上一百个commit
- 若想reset更新的commit，需要知道版本号 git reset [版本号]

## 分支管理

- 分支的切换：HEAD一直指向当前分支最新commit，切换分支时，HEAD指向另一个分支
- git branch: list all branchs
- git branch <>: create branch
- git switch <> / git checkout <>：切换
- git switch -c <> / git checkout -b <>: 创建并切换
- git branch -d <>: 删除
- git merge <new branch>: 合并new branch到当前分支
  - 无冲突：fast forward
  - 有冲突：解决冲突，commit。git log --graph可以看分支合并图
- git rebase:把本地未push的分叉提交历史整理成直线
[rebase 和 merge](https://blog.csdn.net/liuxiaoheng1992/article/details/79108233)

### 分支管理策略

- fast forward：不会留下commit信息
- --no-ff：会留下commit信息，用法：git merge --no-ff -m "merge with no-ff" dev

### 工作现场保护&Bug分支

https://www.liaoxuefeng.com/wiki/896043488029600/900388704535136
- 修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；
- 当手头工作没有完成时，先把工作现场git stash一下，然后去修复bug，修复后，再git stash pop，回到工作现场；
- 在master分支上修复的bug，想要合并到当前dev分支，可以用git cherry-pick <commit>命令，把bug提交的修改“复制”到当前分支，避免重复劳动。

### 多人协作

- 试图用git push origin <your-branch>推送自己的修改；
- 如果推送失败，则因为远程分支比你的本地更新，需要先用git pull试图合并；
  - 如果合并有冲突，则解决冲突，并在本地提交；
  - 如果git pull提示no tracking information，则说明本地分支和远程分支的链接关系没有创建，用命令git branch --set-upstream-to=origin/<origin-branch> <your-branch>
- git push origin <your-branch>

或者：
- git push -u <remote> <branch>，其中u为upstream。git push -u origin branch：把本地branch分支push到origin/branch
- git push --set-upstream <remote> <branch>

## ignore

- [部分规则](https://github.com/github/gitignore)