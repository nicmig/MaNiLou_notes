---
date: April 2025
author: Marc-Antoine Fortin & ChatGPT
email: marc.a.fortin@ntnu.no
github: mafortin
---

# Git Cheat Sheet

## 1) Basic Git Commands

| Command | Description |
|--------|-------------|
| `git init` | Initialize a new git repo |
| `git clone <url>` | Clone an existing repo |
| `git status` | Show changes and tracked files |
| `git add <file>` | Stage a file |
| `git commit -m "msg"` | Commit changes |
| `git log` | Show commit history |
| `git diff` | Show unstaged changes |

### Branching & Merging

| Command | Description |
|--------|-------------|
| `git branch` | List branches |
| `git branch <name>` | Create a new branch |
| `git checkout <name>` | Switch to branch |
| `git checkout -b <name>` | Create & switch to branch |
| `git merge <branch>` | Merge into current branch |
| `git branch -d <name>` | Delete a branch |

### Pushing & Pulling

| Command | Description |
|--------|-------------|
| `git remote -v` | Show remotes |
| `git push origin <branch>` | Push branch to origin |
| `git pull origin <branch>` | Pull changes from origin |

### Sync & Rebase

| Command | Description |
|--------|-------------|
| `git fetch` | Get latest changes from remote |
| `git rebase origin/main` | Rebase current branch on main |
| `git pull --rebase` | Pull using rebase |
| `git stash` | Save changes temporarily |
| `git stash pop` | Re-apply stashed changes |

### Fixing Mistakes

| Command | Description |
|--------|-------------|
| `git commit --amend` | Edit last commit |
| `git reset --soft HEAD~1` | Undo last commit, keep changes staged |
| `git reset --hard HEAD~1` | Undo last commit, discard changes |
| `git revert <commit>` | Revert a commit with a new one |

### Collaboration Tips

- Use **feature branches**: `feature/login`, `bugfix/api-error`
- Pull latest main before branching
- Squash or clean up commits before merging to `main`
- Use **Pull Requests (PRs)** or **Merge Requests (MRs)** for reviews

### Best Practices

- Commit early, commit often
- Write meaningful commit messages
- Don’t push directly to `main` (use PRs)
- Regularly pull and rebase to avoid conflicts

---


## 2) Standard Operating Procedure (SOP) for Merging a secondary branch into the main one



### Step 1: Sync Local Main with Remote Main

Ensure your local `main` branch is up to date with the latest changes from the remote:

```bash
git checkout main
git pull origin main
```

### Step 2: Merge or Rebase Main into Your Feature Branch

Switch to your feature branch (`other_branch_name`) and bring in the latest changes from `main`:

```bash
git checkout other_branch_name
git merge main
# Or, alternatively (for a cleaner history):
# git rebase main
```

Resolve any merge conflicts if necessary, then push the updated branch:

```bash
git push origin other_branch_name
```

### Step 3: Create a Pull Request (PR)

1. Go to your repository hosting service (e.g., GitHub).
2. Navigate to your repository.
3. Look for a prompt to "Compare & pull request" for your `other_branch_name` branch.
4. Create a pull request to merge `other_branch_name` into `main`.
5. Review the changes and resolve any conflicts if needed.
6. Merge the pull request into `main` once everything looks good.

### If There Are Merge Conflicts Online

If GitHub (or other platform) shows merge conflicts that can't be resolved in the UI, handle them locally:

```bash
git checkout other_branch_name
git merge main
```

Resolve conflicts in your code editor, then:

```bash
git add .
git commit -m "Resolved merge conflicts with main"
git push origin other_branch_name
```

Then return to your hosting platform and complete the pull request.

---
