"""
LaaLM-v2 Data Generator v3 - UNAMBIGUOUS FORMAT
Clear delimiters, no hallucination potential
"""

import json
import random
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Configuration
NUM_CONVERSATIONS = 17500
COMMANDS_PER_CONVERSATION = (30, 50)
OUTPUT_FILE = "laalm_v2_training_data_v3.jsonl"

@dataclass
class FileNode:
    name: str
    is_dir: bool
    content: str = ""
    children: Dict[str, 'FileNode'] = field(default_factory=dict)

class FileSystem:
    def __init__(self):
        self.root = FileNode("", is_dir=True)
        self.cwd = "/home/user"
        self._init_home()

    def _init_home(self):
        parts = self.cwd.strip("/").split("/")
        node = self.root
        for part in parts:
            if part not in node.children:
                node.children[part] = FileNode(part, is_dir=True)
            node = node.children[part]

    def _get_node(self, path: str) -> Optional[FileNode]:
        if path == "/":
            return self.root
        parts = [p for p in path.strip("/").split("/") if p]
        node = self.root
        for part in parts:
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def _get_parent_and_name(self, path: str) -> Tuple[Optional[FileNode], str]:
        if path == "/":
            return None, ""
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts:
            return None, ""
        name = parts[-1]
        if len(parts) == 1:
            return self.root, name
        parent_path = "/" + "/".join(parts[:-1])
        return self._get_node(parent_path), name

    def resolve_path(self, path: str) -> str:
        if path == "~":
            return "/home/user"
        if path.startswith("~/"):
            path = "/home/user/" + path[2:]
        if not path.startswith("/"):
            if self.cwd == "/":
                path = "/" + path
            else:
                path = self.cwd + "/" + path

        # Normalize: resolve . and .. components
        parts = path.split("/")
        resolved = []
        for part in parts:
            if part == "" or part == ".":
                continue
            elif part == "..":
                if resolved:
                    resolved.pop()
            else:
                resolved.append(part)
        return "/" + "/".join(resolved) if resolved else "/"

    def list_dir(self, path: str = ".") -> Optional[List[str]]:
        abs_path = self.resolve_path(path)
        node = self._get_node(abs_path)
        if not node or not node.is_dir:
            return None
        return sorted(node.children.keys())

    def create_file(self, path: str, content: str = "") -> bool:
        abs_path = self.resolve_path(path)
        parent, name = self._get_parent_and_name(abs_path)
        if not parent or not parent.is_dir:
            return False
        parent.children[name] = FileNode(name, is_dir=False, content=content)
        return True

    def create_dir(self, path: str, parents: bool = False) -> bool:
        abs_path = self.resolve_path(path)
        if parents:
            # mkdir -p: create all intermediate directories
            parts = [p for p in abs_path.strip("/").split("/") if p]
            node = self.root
            for part in parts:
                if part not in node.children:
                    node.children[part] = FileNode(part, is_dir=True)
                elif not node.children[part].is_dir:
                    return False  # path component is a file
                node = node.children[part]
            return True
        else:
            parent, name = self._get_parent_and_name(abs_path)
            if not parent or not parent.is_dir:
                return False
            if name in parent.children:
                return False
            parent.children[name] = FileNode(name, is_dir=True)
            return True

    def read_file(self, path: str) -> Optional[str]:
        abs_path = self.resolve_path(path)
        node = self._get_node(abs_path)
        if not node or node.is_dir:
            return None
        return node.content

    def write_file(self, path: str, content: str, append: bool = False) -> bool:
        abs_path = self.resolve_path(path)
        node = self._get_node(abs_path)
        if node:
            if node.is_dir:
                return False
            if append:
                node.content += content
            else:
                node.content = content
            return True
        else:
            return self.create_file(path, content)

    def delete(self, path: str, recursive: bool = False) -> bool:
        abs_path = self.resolve_path(path)
        parent, name = self._get_parent_and_name(abs_path)
        if not parent or name not in parent.children:
            return False
        node = parent.children[name]
        if node.is_dir and not recursive:
            return False
        del parent.children[name]
        return True

    def move(self, src: str, dst: str) -> bool:
        src_abs = self.resolve_path(src)
        dst_abs = self.resolve_path(dst)
        src_node = self._get_node(src_abs)
        if not src_node:
            return False
        src_parent, src_name = self._get_parent_and_name(src_abs)
        if not src_parent or src_name not in src_parent.children:
            return False

        # If dst is an existing directory, move src into it
        dst_node = self._get_node(dst_abs)
        if dst_node and dst_node.is_dir:
            dst_node.children[src_name] = src_node
            del src_parent.children[src_name]
            return True

        # Otherwise treat dst as the new name/path
        dst_parent, dst_name = self._get_parent_and_name(dst_abs)
        if not dst_parent or not dst_parent.is_dir:
            return False
        del src_parent.children[src_name]
        src_node.name = dst_name
        dst_parent.children[dst_name] = src_node
        return True

    def copy(self, src: str, dst: str) -> bool:
        src_abs = self.resolve_path(src)
        dst_abs = self.resolve_path(dst)
        src_node = self._get_node(src_abs)
        if not src_node or src_node.is_dir:
            return False

        # If dst is an existing directory, copy into it
        dst_node = self._get_node(dst_abs)
        if dst_node and dst_node.is_dir:
            new_node = FileNode(src_node.name, is_dir=False, content=src_node.content)
            dst_node.children[src_node.name] = new_node
            return True

        dst_parent, dst_name = self._get_parent_and_name(dst_abs)
        if not dst_parent or not dst_parent.is_dir:
            return False
        new_node = FileNode(dst_name, is_dir=False, content=src_node.content)
        dst_parent.children[dst_name] = new_node
        return True

    def exists(self, path: str) -> bool:
        abs_path = self.resolve_path(path)
        return self._get_node(abs_path) is not None

    def is_directory(self, path: str) -> bool:
        abs_path = self.resolve_path(path)
        node = self._get_node(abs_path)
        return node is not None and node.is_dir

    def change_dir(self, path: str) -> bool:
        if path == "~":
            path = "/home/user"
        abs_path = self.resolve_path(path)
        node = self._get_node(abs_path)
        if not node or not node.is_dir:
            return False
        self.cwd = abs_path
        return True

# Random generators
def random_filename(extension: str = "") -> str:
    prefixes = ["test", "data", "file", "doc", "report", "output", "temp", "backup", "config", "log"]
    name = random.choice(prefixes)
    if random.random() > 0.5:
        name += "_" + "".join(random.choices(string.digits, k=random.randint(1, 3)))
    if extension:
        name += extension
    elif random.random() > 0.6:
        ext_choices = [".txt", ".log", ".dat", ".csv", ".json", ".md"]
        name += random.choice(ext_choices)
    return name

def random_dirname() -> str:
    names = ["docs", "data", "tmp", "backup", "config", "logs", "output", "projects", "files", "archives"]
    name = random.choice(names)
    if random.random() > 0.6:
        name += "_" + "".join(random.choices(string.digits, k=random.randint(1, 2)))
    return name

def random_content(lines: int = None) -> str:
    if lines is None:
        lines = random.randint(1, 10)
    words_pool = [
        "hello", "world", "test", "data", "error", "warning", "info", "success",
        "failed", "processing", "complete", "started", "finished", "running",
        "system", "user", "admin", "log", "output", "input", "result"
    ]
    content_lines = []
    for i in range(lines):
        line_words = random.sample(words_pool, k=random.randint(2, 6))
        content_lines.append(" ".join(line_words))
    return "\n".join(content_lines)

def _strip_quotes(text: str) -> str:
    """Strip surrounding quotes from a string, like bash does."""
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            return text[1:-1]
    return text

# Command executor
class CommandExecutor:
    def __init__(self, fs: FileSystem):
        self.fs = fs

    def execute(self, command: str) -> Tuple[str, Optional[str]]:
        """Execute and return (output, reasoning)"""
        command = command.strip()

        if "|" in command:
            return self._execute_pipe(command)

        if ">>" in command:
            return self._execute_append_redirect(command)

        parts = command.split()
        if not parts:
            return "", None

        cmd = parts[0]
        args = parts[1:]

        handlers = {
            "pwd": self._pwd,
            "ls": self._ls,
            "cd": self._cd,
            "touch": self._touch,
            "mkdir": self._mkdir,
            "rm": self._rm,
            "mv": self._mv,
            "cp": self._cp,
            "cat": self._cat,
            "echo": self._echo,
            "grep": self._grep,
            "head": self._head,
            "tail": self._tail,
            "wc": self._wc,
            "find": self._find,
        }

        if cmd not in handlers:
            return f"bash: {cmd}: command not found", None

        return handlers[cmd](args)

    def _execute_pipe(self, command: str) -> Tuple[str, str]:
        parts = [p.strip() for p in command.split("|")]
        reasoning = "<REASON>\n"
        current_output = ""

        for i, part in enumerate(parts):
            step_num = i + 1
            if i == 0:
                output, _ = self.execute(part)
                current_output = output
                reasoning += f"STEP{step_num}: execute({part}) -> output=\"{output[:50]}{'...' if len(output) > 50 else ''}\"\n"
            else:
                reasoning += f"STEP{step_num}: pipe_to({part}, input=\"{current_output[:50]}{'...' if len(current_output) > 50 else ''}\")\n"
                cmd_parts = part.split()
                if cmd_parts[0] == "grep" and len(cmd_parts) > 1:
                    pattern = cmd_parts[1]
                    filtered = [line for line in current_output.split("\n") if pattern in line]
                    current_output = "\n".join(filtered)
                    reasoning += f"  -> filtered={len(filtered)} lines\n"
                elif cmd_parts[0] == "head":
                    n = 10
                    if len(cmd_parts) > 1:
                        if cmd_parts[1] == "-n" and len(cmd_parts) > 2 and cmd_parts[2].isdigit():
                            n = int(cmd_parts[2])
                        elif cmd_parts[1].lstrip("-").isdigit():
                            n = int(cmd_parts[1].lstrip("-"))
                    lines = current_output.split("\n")[:n]
                    current_output = "\n".join(lines)
                    reasoning += f"  -> first {n} lines\n"
                elif cmd_parts[0] == "tail":
                    n = 10
                    if len(cmd_parts) > 1:
                        if cmd_parts[1] == "-n" and len(cmd_parts) > 2 and cmd_parts[2].isdigit():
                            n = int(cmd_parts[2])
                        elif cmd_parts[1].lstrip("-").isdigit():
                            n = int(cmd_parts[1].lstrip("-"))
                    lines = current_output.split("\n")[-n:]
                    current_output = "\n".join(lines)
                    reasoning += f"  -> last {n} lines\n"
                elif cmd_parts[0] == "wc":
                    lines = len(current_output.split("\n")) if current_output else 0
                    words = len(current_output.split()) if current_output else 0
                    chars = len(current_output)
                    current_output = f"{lines} {words} {chars}"
                    reasoning += f"  -> {lines}L {words}W {chars}C\n"
                else:
                    current_output = f"bash: {cmd_parts[0]}: cannot use in pipe"

        reasoning += "</REASON>"
        return current_output, reasoning

    def _execute_append_redirect(self, command: str) -> Tuple[str, None]:
        parts = command.split(">>")
        if len(parts) != 2:
            return "bash: syntax error near unexpected token `>>'", None
        cmd_part = parts[0].strip()
        file_part = parts[1].strip()
        output, _ = self.execute(cmd_part)
        # Real >> appends output with a trailing newline
        if self.fs.write_file(file_part, output + "\n", append=True):
            return "", None
        else:
            return f"bash: {file_part}: No such file or directory", None

    def _pwd(self, args: List[str]) -> Tuple[str, None]:
        return self.fs.cwd, None

    def _ls(self, args: List[str]) -> Tuple[str, None]:
        path = args[0] if args else "."
        files = self.fs.list_dir(path)
        if files is None:
            if not self.fs.exists(path):
                return f"ls: cannot access '{path}': No such file or directory", None
            else:
                return f"ls: cannot access '{path}': Not a directory", None
        if not files:
            return "", None
        return "\n".join(files), None

    def _cd(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            self.fs.cwd = "/home/user"
            return "", None
        path = args[0]
        if self.fs.change_dir(path):
            return "", None
        else:
            if not self.fs.exists(path):
                return f"cd: {path}: No such file or directory", None
            else:
                return f"cd: {path}: Not a directory", None

    def _touch(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            return "touch: missing file operand", None
        for path in args:
            if not self.fs.exists(path):
                self.fs.create_file(path)
        return "", None

    def _mkdir(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            return "mkdir: missing operand", None
        # Check for -p flag
        parents = False
        paths = []
        for arg in args:
            if arg == "-p":
                parents = True
            else:
                paths.append(arg)
        if not paths:
            return "mkdir: missing operand", None
        results = []
        for path in paths:
            if not parents and self.fs.exists(path):
                results.append(f"mkdir: cannot create directory '{path}': File exists")
            elif self.fs.create_dir(path, parents=parents):
                pass  # success, no output
            else:
                results.append(f"mkdir: cannot create directory '{path}': No such file or directory")
        return "\n".join(results), None

    def _rm(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            return "rm: missing operand", None
        # Check for -r/-rf flags
        recursive = False
        paths = []
        for arg in args:
            if arg in ("-r", "-rf", "-fr"):
                recursive = True
            elif arg.startswith("-") and "r" in arg:
                recursive = True
            else:
                paths.append(arg)
        if not paths:
            return "rm: missing operand", None
        results = []
        for path in paths:
            if not self.fs.exists(path):
                results.append(f"rm: cannot remove '{path}': No such file or directory")
            elif self.fs.is_directory(path) and not recursive:
                results.append(f"rm: cannot remove '{path}': Is a directory")
            else:
                self.fs.delete(path, recursive=recursive)
        return "\n".join(results), None

    def _mv(self, args: List[str]) -> Tuple[str, None]:
        if len(args) < 2:
            return "mv: missing file operand", None
        src, dst = args[0], args[1]
        if not self.fs.exists(src):
            return f"mv: cannot stat '{src}': No such file or directory", None
        if self.fs.move(src, dst):
            return "", None
        else:
            return f"mv: cannot move '{src}' to '{dst}': No such file or directory", None

    def _cp(self, args: List[str]) -> Tuple[str, None]:
        if len(args) < 2:
            return "cp: missing file operand", None
        src, dst = args[0], args[1]
        if not self.fs.exists(src):
            return f"cp: cannot stat '{src}': No such file or directory", None
        if self.fs.is_directory(src):
            return f"cp: -r not specified; omitting directory '{src}'", None
        if self.fs.copy(src, dst):
            return "", None
        else:
            return f"cp: cannot create regular file '{dst}': No such file or directory", None

    def _cat(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            return "cat: missing file operand", None
        path = args[0]
        content = self.fs.read_file(path)
        if content is None:
            if not self.fs.exists(path):
                return f"cat: {path}: No such file or directory", None
            else:
                return f"cat: {path}: Is a directory", None
        return content, None

    def _echo(self, args: List[str]) -> Tuple[str, None]:
        if ">" in args:
            idx = args.index(">")
            text = " ".join(args[:idx])
            text = _strip_quotes(text)
            if idx + 1 >= len(args):
                return "bash: syntax error near unexpected token `newline'", None
            filename = args[idx + 1]
            self.fs.write_file(filename, text)
            return "", None
        text = " ".join(args)
        text = _strip_quotes(text)
        return text, None

    def _grep(self, args: List[str]) -> Tuple[str, None]:
        if len(args) < 2:
            return "grep: missing operand", None
        pattern = _strip_quotes(args[0])
        filename = args[1]
        content = self.fs.read_file(filename)
        if content is None:
            if not self.fs.exists(filename):
                return f"grep: {filename}: No such file or directory", None
            else:
                return f"grep: {filename}: Is a directory", None
        matching_lines = [line for line in content.split("\n") if pattern in line]
        return "\n".join(matching_lines), None

    def _head(self, args: List[str]) -> Tuple[str, None]:
        n = 10
        filename = None
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args):
                try:
                    n = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i].startswith("-") and args[i][1:].isdigit():
                n = int(args[i][1:])
                i += 1
            else:
                filename = args[i]
                i += 1
        if not filename:
            return "head: missing file operand", None
        content = self.fs.read_file(filename)
        if content is None:
            if not self.fs.exists(filename):
                return f"head: cannot open '{filename}' for reading: No such file or directory", None
            else:
                return f"head: error reading '{filename}': Is a directory", None
        lines = content.split("\n")[:n]
        return "\n".join(lines), None

    def _tail(self, args: List[str]) -> Tuple[str, None]:
        n = 10
        filename = None
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args):
                try:
                    n = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i].startswith("-") and args[i][1:].isdigit():
                n = int(args[i][1:])
                i += 1
            else:
                filename = args[i]
                i += 1
        if not filename:
            return "tail: missing file operand", None
        content = self.fs.read_file(filename)
        if content is None:
            if not self.fs.exists(filename):
                return f"tail: cannot open '{filename}' for reading: No such file or directory", None
            else:
                return f"tail: error reading '{filename}': Is a directory", None
        lines = content.split("\n")[-n:]
        return "\n".join(lines), None

    def _wc(self, args: List[str]) -> Tuple[str, None]:
        if not args:
            return "wc: missing file operand", None
        filename = args[0]
        content = self.fs.read_file(filename)
        if content is None:
            if not self.fs.exists(filename):
                return f"wc: {filename}: No such file or directory", None
            else:
                return f"wc: {filename}: Is a directory", None
        lines = len(content.split("\n")) if content else 0
        words = len(content.split()) if content else 0
        chars = len(content)
        return f"{lines} {words} {chars} {filename}", None

    def _find(self, args: List[str]) -> Tuple[str, str]:
        path = args[0] if args else "."
        reasoning = "<REASON>\n"
        reasoning += f"STEP1: traverse({path})\n"
        results = []

        def traverse(current_path: str, depth: int = 0):
            if depth > 10:
                return
            node = self.fs._get_node(current_path)
            if not node:
                return
            results.append(current_path)
            if node.is_dir:
                for child_name in sorted(node.children.keys()):
                    child_path = f"{current_path}/{child_name}" if current_path != "/" else f"/{child_name}"
                    traverse(child_path, depth + 1)

        abs_path = self.fs.resolve_path(path)
        if not self.fs.exists(abs_path):
            return f"find: '{path}': No such file or directory", None
        traverse(abs_path)
        reasoning += f"STEP2: found {len(results)} items\n"
        reasoning += "</REASON>"
        return "\n".join(results), reasoning

# ============================================================================
# CONVERSATION GENERATOR - UNAMBIGUOUS FORMAT
# ============================================================================

def generate_conversation() -> str:
    """Generate conversation with CRYSTAL CLEAR format"""
    fs = FileSystem()
    executor = CommandExecutor(fs)

    # Start conversation
    conversation_text = "### SYSTEM ###\nCWD=/home/user\nFILES=[]\nENV=USER:user,HOME:/home/user\n### END SYSTEM ###\n\n"

    num_commands = random.randint(*COMMANDS_PER_CONVERSATION)

    command_pool = {
        "create_file": 15, "create_dir": 10, "list": 12, "read_file": 10,
        "write_to_file": 8, "append_to_file": 4, "move_file": 5, "copy_file": 5,
        "delete_file": 5, "change_dir": 8, "pwd": 6, "grep_file": 6,
        "head_file": 4, "tail_file": 4, "wc_file": 3, "pipe_grep": 6,
        "pipe_head": 4, "pipe_tail": 4, "pipe_wc": 3, "find": 4,
        "mkdir_p": 3, "rm_rf": 3,
        "error_no_file": 8, "error_wrong_type": 4,
    }

    for _ in range(num_commands):
        action = random.choices(list(command_pool.keys()), weights=list(command_pool.values()))[0]
        command = None

        if action == "create_file":
            filename = random_filename()
            command = f"touch {filename}"
        elif action == "create_dir":
            dirname = random_dirname()
            command = f"mkdir {dirname}"
        elif action == "mkdir_p":
            d1 = random_dirname()
            d2 = random_dirname()
            command = f"mkdir -p {d1}/{d2}"
        elif action == "list":
            dirs = fs.list_dir() or []
            path = random.choice(dirs) if dirs and random.random() > 0.5 else "."
            command = f"ls {path}" if path != "." else "ls"
        elif action == "read_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"cat {random.choice(files)}"
        elif action == "write_to_file":
            filename = random_filename()
            content = random.choice(["hello", "test data", "error log", "success", "processing"])
            command = f"echo {content} > {filename}"
        elif action == "append_to_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                filename = random.choice(files)
                content = random.choice(["more data", "additional info", "update", "new entry"])
                command = f"echo {content} >> {filename}"
        elif action == "move_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            if files:
                src = random.choice(files)
                # Sometimes move into a directory, sometimes rename
                if dirs and random.random() > 0.5:
                    command = f"mv {src} {random.choice(dirs)}"
                else:
                    command = f"mv {src} {random_filename()}"
        elif action == "copy_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            if files:
                src = random.choice(files)
                if dirs and random.random() > 0.5:
                    command = f"cp {src} {random.choice(dirs)}"
                else:
                    command = f"cp {src} {random_filename()}"
        elif action == "delete_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"rm {random.choice(files)}"
        elif action == "rm_rf":
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            if dirs:
                command = f"rm -rf {random.choice(dirs)}"
        elif action == "change_dir":
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            if dirs and random.random() > 0.3:
                command = f"cd {random.choice(dirs)}"
            elif random.random() > 0.5:
                command = "cd .."
            else:
                command = "cd"
        elif action == "pwd":
            command = "pwd"
        elif action == "grep_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                pattern = random.choice(["error", "test", "data", "info", "warning"])
                command = f"grep {pattern} {random.choice(files)}"
        elif action == "head_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"head {random.choice(files)}"
        elif action == "tail_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"tail {random.choice(files)}"
        elif action == "wc_file":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"wc {random.choice(files)}"
        elif action == "pipe_grep":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                pattern = random.choice(["error", "test", "data", "info"])
                command = f"cat {random.choice(files)} | grep {pattern}"
        elif action == "pipe_head":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"cat {random.choice(files)} | head"
        elif action == "pipe_tail":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"cat {random.choice(files)} | tail"
        elif action == "pipe_wc":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            if files:
                command = f"cat {random.choice(files)} | wc"
        elif action == "find":
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            path = random.choice(dirs) if dirs and random.random() > 0.5 else "."
            command = f"find {path}"
        elif action == "error_no_file":
            # Use a name unlikely to exist in the filesystem
            fake_file = "nonexistent_" + "".join(random.choices(string.ascii_lowercase, k=6))
            cmd_choice = random.choice(["cat", "rm", "grep error", "head", "tail", "wc"])
            command = f"{cmd_choice} {fake_file}"
        elif action == "error_wrong_type":
            files = [f for f in (fs.list_dir() or []) if not fs.is_directory(f)]
            dirs = [d for d in (fs.list_dir() or []) if fs.is_directory(d)]
            if files and random.random() > 0.5:
                command = f"cd {random.choice(files)}"
            elif dirs:
                command = f"cat {random.choice(dirs)}"

        if command:
            output, reasoning = executor.execute(command)

            # UNAMBIGUOUS FORMAT:
            conversation_text += f"### COMMAND ###\n{command}\n### END COMMAND ###\n\n"

            if reasoning:
                conversation_text += f"### OUTPUT ###\n{reasoning}\n"
                if output:
                    conversation_text += f"{output}\n"
                conversation_text += "### END OUTPUT ###\n\n"
            else:
                conversation_text += f"### OUTPUT ###\n{output}\n### END OUTPUT ###\n\n"

    return conversation_text

def main():
    print(f"Generating {NUM_CONVERSATIONS} conversations with UNAMBIGUOUS format...")
    print(f"Commands per conversation: {COMMANDS_PER_CONVERSATION[0]}-{COMMANDS_PER_CONVERSATION[1]}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with open(OUTPUT_FILE, "w") as f:
        for i in range(NUM_CONVERSATIONS):
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{NUM_CONVERSATIONS} conversations...")

            conversation = generate_conversation()
            # Save as single line JSON with text field
            f.write(json.dumps({"text": conversation}) + "\n")

    print(f"\nDone! Generated {NUM_CONVERSATIONS} conversations")
    print(f"Saved to: {OUTPUT_FILE}")

    # Show example
    print("\n" + "="*60)
    print("EXAMPLE CONVERSATION FORMAT:")
    print("="*60)
    with open(OUTPUT_FILE) as f:
        example = json.loads(f.readline())
        print(example["text"][:800])
        print("\n[... truncated ...]")

if __name__ == "__main__":
    main()
