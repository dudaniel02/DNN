import git
import os
import subprocess

# Set repository path
repo_path = r"C:\Users\dhber\School 4\DNN Exam"
repo = git.Repo(repo_path)

# Identify large files (over 50 MB)
def get_large_files(repo, size_limit):
    large_files = []
    for item in repo.tree().traverse():
        if item.type == 'blob' and item.size > size_limit:
            large_files.append((item.path, item.size))
    return large_files

size_limit = 50 * 1024 * 1024  # 50 MB
large_files = get_large_files(repo, size_limit)

print("Large files:")
for path, size in large_files:
    print(f"{path} - {size / (1024 * 1024):.2f} MB")

# Remove large files from index and working tree
files_to_remove = [path for path, size in large_files]
if files_to_remove:
    repo.index.remove(files_to_remove, working_tree=True)

# Update .gitignore
gitignore_path = os.path.join(repo_path, '.gitignore')
ignore_patterns = [
    'env_pytorch/',
    'outputs/checkpoints/',
    '*.dll',
    '*.lib',
    '*.pyd',
    '*.keras'
]

with open(gitignore_path, 'a') as gitignore:
    for pattern in ignore_patterns:
        gitignore.write(pattern + '\n')

# Stage and commit changes
repo.git.add('.gitignore')
repo.index.commit("Remove large files and update .gitignore")

# Remove large files from Git history using BFG Repo-Cleaner
bfg_jar_path = r"C:\bfg\bfg.jar"
delete_patterns = "*.keras,*.dll,*.lib,*.pyd"

subprocess.run([
    'java', '-jar', bfg_jar_path,
    '--delete-files', delete_patterns,
    repo_path
])

# Clean up the repository
subprocess.run(['git', 'reflog', 'expire', '--expire=now', '--all'], cwd=repo_path)
subprocess.run(['git', 'gc', '--prune=now', '--aggressive'], cwd=repo_path)

# Force push to remote repository
origin = repo.remote(name='origin')
origin.push(refspec='main:main', force=True)
