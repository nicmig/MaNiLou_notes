---
date: April 2025
author: Marc-Antoine Fortin & ChatGPT
email: marc.a.fortin@ntnu.no
github: mafortin
---
# Git Cheat Sheet

## Basic Commands

| Command | Description |
|--------|-------------|
| `git init` | Initialize a new git repo |
| `git clone <url>` | Clone an existing repo |
| `git status` | Show changes and tracked files |
| `git add <file>` | Stage a file |
| `git commit -m "msg"` | Commit changes |
| `git log` | Show commit history |
| `git diff` | Show unstaged changes |

## Branching & Merging

| Command | Description |
|--------|-------------|
| `git branch` | List branches |
| `git branch <name>` | Create a new branch |
| `git checkout <name>` | Switch to branch |
| `git checkout -b <name>` | Create & switch to branch |
| `git merge <branch>` | Merge into current branch |
| `git branch -d <name>` | Delete a branch |

## Pushing & Pulling

| Command | Description |
|--------|-------------|
| `git remote -v` | Show remotes |
| `git push origin <branch>` | Push branch to origin |
| `git pull origin <branch>` | Pull changes from origin |

## Sync & Rebase

| Command | Description |
|--------|-------------|
| `git fetch` | Get latest changes from remote |
| `git rebase origin/main` | Rebase current branch on main |
| `git pull --rebase` | Pull using rebase |
| `git stash` | Save changes temporarily |
| `git stash pop` | Re-apply stashed changes |

## Fixing Mistakes

| Command | Description |
|--------|-------------|
| `git commit --amend` | Edit last commit |
| `git reset --soft HEAD~1` | Undo last commit, keep changes staged |
| `git reset --hard HEAD~1` | Undo last commit, discard changes |
| `git revert <commit>` | Revert a commit with a new one |

## Collaboration Tips

- Use **feature branches**: `feature/login`, `bugfix/api-error`
- Pull latest main before branching
- Squash or clean up commits before merging to `main`
- Use **Pull Requests (PRs)** or **Merge Requests (MRs)** for reviews

## Best Practices

- Commit early, commit often
- Write meaningful commit messages
- Don’t push directly to `main` (use PRs)
- Regularly pull and rebase to avoid conflicts

---

Happy coding!
