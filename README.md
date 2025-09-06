# Go Code-Editing Agent with Groq Integration

A powerful AI-powered code-editing agent that can read, write, and manipulate files and directories, plus execute terminal commands. Built with Go and powered by Groq's lightning-fast LPU infrastructure using Llama 3.3 70B.

## 🚀 Features

### 📁 File Operations
- **read_file** - Read contents of any file
- **create_file** - Create new files with specified content
- **edit_file** - Edit files using string replacement
- **delete_file** - Delete existing files
- **rename_file** - Rename or move files

### 📂 Directory Operations  
- **list_files** - List files and directories recursively
- **create_folder** - Create new directories
- **delete_folder** - Delete directories and all contents
- **rename_folder** - Rename or move directories

### 💻 Terminal Operations
- **terminal_run** - Execute terminal commands and capture output

### 🌐 Web Development
- **create_website** - Create complete websites with HTML, CSS, and JavaScript files

## 🛠️ Setup

1. **Get a Groq API key** from [console.groq.com](https://console.groq.com)

2. **Set environment variable**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   # Windows PowerShell:
   $env:GROQ_API_KEY = "your-groq-api-key-here"
   ```

3. **Run the agent**:
   ```bash
   go run main.go
   # OR build and run:
   go build -o agent
   ./agent
   ```

## 🎯 Example Usage

```
You: create a new Python script called hello.py
Agent: I'll create a Python script for you.
[Uses create_file tool]

You: what files do you see in this directory?
Agent: Let me check the current directory.
[Uses list_files tool]

You: run the Python script
Agent: I'll execute the Python script for you.
[Uses terminal_run tool]

You: create a new folder called "projects"
Agent: I'll create the projects folder.
[Uses create_folder tool]
```

## ⚡ Powered by Groq

- **Ultra-fast inference** with Groq's LPU technology
- **Llama 3.3 70B** model for powerful reasoning
- **OpenAI-compatible API** for seamless integration

## 🔧 Tools Reference

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path` |
| `create_file` | Create new file | `path`, `content` |
| `edit_file` | Edit file via replacement | `path`, `old_str`, `new_str` |
| `delete_file` | Delete file | `path` |
| `rename_file` | Rename/move file | `old_path`, `new_path` |
| `list_files` | List directory contents | `path` (optional) |
| `create_folder` | Create directory | `path` |
| `delete_folder` | Delete directory | `path` |
| `rename_folder` | Rename/move directory | `old_path`, `new_path` |
| `terminal_run` | Execute command | `command`, `timeout` (optional) |
| `create_website` | Create complete website | `folder_path`, `project_name`, `description`, `style` (optional) |

## ⚠️ Safety Notes

- **delete_file** and **delete_folder** operations cannot be undone
- **terminal_run** can execute any system command - use with caution
- The agent has full access to your file system within the working directory

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
