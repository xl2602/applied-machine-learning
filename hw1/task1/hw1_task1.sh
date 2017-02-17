#!/bin/sh

commit_new_file(){
if [ "$#" -eq 2 ]; then
 for ((i=$1;i<=$2;i++)); do
  touch $i
  git add $i
  git commit -m $i
 done
else
  echo "only commit for " $1 
  touch $1
  git add $1
  git commit -m $1
fi
}

mkdir task1
cd task1
git init

commit_new_file 1 5

git checkout -b feature HEAD~4

commit_new_file 6 8

git rebase --onto feature master~2 master
git checkout master@{3}

commit_new_file 9

git checkout -b debug

git rebase --onto debug feature~2 feature~1
git rebase HEAD debug
git reset --soft HEAD^
git commit add 7
git commit --amend -m "7+9"

