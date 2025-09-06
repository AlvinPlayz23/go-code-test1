package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/invopop/jsonschema"
)

type ToolDefinition struct {
	Name        string                     `json:"name"`
	Description string                     `json:"description"`
	InputSchema any                        `json:"parameters"`
	Function    func(input json.RawMessage) (string, error)
}

func main() {
	config := openai.DefaultConfig(os.Getenv("GROQ_API_KEY"))
	config.BaseURL = "https://api.groq.com/openai/v1"
	client := openai.NewClientWithConfig(config)

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{
		ReadFileDefinition,
		ListFilesDefinition,
		EditFileDefinition,
		CreateFileDefinition,
		DeleteFileDefinition,
		RenameFileDefinition,
		CreateFolderDefinition,
		DeleteFolderDefinition,
		RenameFolderDefinition,
		TerminalRunDefinition,
		CreateWebsiteDefinition,
	}
	agent := NewAgent(client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(
	client *openai.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessage{}

	fmt.Println("Chat with Groq (use 'ctrl-c' to quit)")

	readUserInput := true
	for {
		if readUserInput {
			fmt.Print("\u001b[94mYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			userMessage := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: userInput,
			}
			conversation = append(conversation, userMessage)
		}

		response, err := a.runInference(ctx, conversation)
		if err != nil {
			fmt.Printf("\u001b[91mError\u001b[0m: %s\n", err.Error())
			readUserInput = true
			continue
		}

		assistantMessage := response.Choices[0].Message
		conversation = append(conversation, assistantMessage)

		// Handle tool calls
		if len(assistantMessage.ToolCalls) > 0 {
			for _, toolCall := range assistantMessage.ToolCalls {
				result := a.executeTool(toolCall.ID, toolCall.Function.Name, json.RawMessage(toolCall.Function.Arguments))
				toolMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    result,
					ToolCallID: toolCall.ID,
				}
				conversation = append(conversation, toolMessage)
			}
			readUserInput = false
		} else {
			fmt.Printf("\u001b[93mGroq\u001b[0m: %s\n", assistantMessage.Content)
			readUserInput = true
		}
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []openai.ChatCompletionMessage) (openai.ChatCompletionResponse, error) {
	tools := []openai.Tool{}
	for _, tool := range a.tools {
		tools = append(tools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	response, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:      "llama-3.3-70b-versatile", // Using Llama 3.3 70B on Groq
		Messages:   conversation,
		Tools:      tools,
		ToolChoice: "auto",
		MaxTokens:  4000, // Increased token limit for longer responses
	})
	return response, err
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) string {
	var toolDef ToolDefinition
	var found bool
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return "tool not found"
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return err.Error()
	}
	return response
}

func GenerateSchema[T any]() any {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)

	// Ensure required is always an array, even if empty
	required := schema.Required
	if required == nil {
		required = []string{}
	}

	return map[string]any{
		"type":       "object",
		"properties": schema.Properties,
		"required":   required,
	}
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema:"required" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The relative path of a file in the working directory.",
		},
	},
	"required": []string{"path"},
}

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "Optional relative path to list files from. Defaults to current directory if not provided.",
		},
	},
	"required": []string{}, // path is optional
}

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		panic(err)
	}

	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: EditFileInputSchema,
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema:"required" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema:"required" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema:"required" jsonschema_description:"Text to replace old_str with"`
}

var EditFileInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The path to the file",
		},
		"old_str": map[string]any{
			"type":        "string",
			"description": "Text to search for - must match exactly and must only have one match exactly",
		},
		"new_str": map[string]any{
			"type":        "string",
			"description": "Text to replace old_str with",
		},
	},
	"required": []string{"path", "old_str", "new_str"},
}

func EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		return "", err
	}

	if editFileInput.Path == "" || editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, -1)

	if oldContent == newContent && editFileInput.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(editFileInput.Path, []byte(newContent), 0644)
	if err != nil {
		return "", err
	}

	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}

var CreateFileDefinition = ToolDefinition{
	Name:        "create_file",
	Description: "Create a new file with specified content. If the file already exists, it will be overwritten.",
	InputSchema: CreateFileInputSchema,
	Function:    CreateFile,
}

type CreateFileInput struct {
	Path    string `json:"path" jsonschema_description:"The path where the file should be created"`
	Content string `json:"content" jsonschema_description:"The content to write to the file"`
}

var CreateFileInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The path where the file should be created",
		},
		"content": map[string]any{
			"type":        "string",
			"description": "The content to write to the file",
		},
	},
	"required": []string{"path", "content"},
}

func CreateFile(input json.RawMessage) (string, error) {
	createFileInput := CreateFileInput{}
	err := json.Unmarshal(input, &createFileInput)
	if err != nil {
		return "", err
	}

	if createFileInput.Path == "" {
		return "", fmt.Errorf("path cannot be empty")
	}

	// Create directory if it doesn't exist
	dir := path.Dir(createFileInput.Path)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err = os.WriteFile(createFileInput.Path, []byte(createFileInput.Content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", createFileInput.Path), nil
}

var DeleteFileDefinition = ToolDefinition{
	Name:        "delete_file",
	Description: "Delete an existing file. Use with caution as this action cannot be undone.",
	InputSchema: DeleteFileInputSchema,
	Function:    DeleteFile,
}

type DeleteFileInput struct {
	Path string `json:"path" jsonschema_description:"The path of the file to delete"`
}

var DeleteFileInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The path of the file to delete",
		},
	},
	"required": []string{"path"},
}

func DeleteFile(input json.RawMessage) (string, error) {
	deleteFileInput := DeleteFileInput{}
	err := json.Unmarshal(input, &deleteFileInput)
	if err != nil {
		return "", err
	}

	if deleteFileInput.Path == "" {
		return "", fmt.Errorf("path cannot be empty")
	}

	// Check if file exists
	if _, err := os.Stat(deleteFileInput.Path); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", deleteFileInput.Path)
	}

	err = os.Remove(deleteFileInput.Path)
	if err != nil {
		return "", fmt.Errorf("failed to delete file: %w", err)
	}

	return fmt.Sprintf("Successfully deleted file %s", deleteFileInput.Path), nil
}

var RenameFileDefinition = ToolDefinition{
	Name:        "rename_file",
	Description: "Rename or move a file from one location to another.",
	InputSchema: RenameFileInputSchema,
	Function:    RenameFile,
}

type RenameFileInput struct {
	OldPath string `json:"old_path" jsonschema_description:"The current path of the file"`
	NewPath string `json:"new_path" jsonschema_description:"The new path for the file"`
}

var RenameFileInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"old_path": map[string]any{
			"type":        "string",
			"description": "The current path of the file",
		},
		"new_path": map[string]any{
			"type":        "string",
			"description": "The new path for the file",
		},
	},
	"required": []string{"old_path", "new_path"},
}

func RenameFile(input json.RawMessage) (string, error) {
	renameFileInput := RenameFileInput{}
	err := json.Unmarshal(input, &renameFileInput)
	if err != nil {
		return "", err
	}

	if renameFileInput.OldPath == "" || renameFileInput.NewPath == "" {
		return "", fmt.Errorf("both old_path and new_path must be provided")
	}

	// Check if source file exists
	if _, err := os.Stat(renameFileInput.OldPath); os.IsNotExist(err) {
		return "", fmt.Errorf("source file does not exist: %s", renameFileInput.OldPath)
	}

	// Create directory for new path if needed
	dir := path.Dir(renameFileInput.NewPath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err = os.Rename(renameFileInput.OldPath, renameFileInput.NewPath)
	if err != nil {
		return "", fmt.Errorf("failed to rename file: %w", err)
	}

	return fmt.Sprintf("Successfully renamed %s to %s", renameFileInput.OldPath, renameFileInput.NewPath), nil
}

var CreateFolderDefinition = ToolDefinition{
	Name:        "create_folder",
	Description: "Create a new directory/folder. Creates parent directories if they don't exist.",
	InputSchema: CreateFolderInputSchema,
	Function:    CreateFolder,
}

type CreateFolderInput struct {
	Path string `json:"path" jsonschema_description:"The path of the directory to create"`
}

var CreateFolderInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The path of the directory to create",
		},
	},
	"required": []string{"path"},
}

func CreateFolder(input json.RawMessage) (string, error) {
	createFolderInput := CreateFolderInput{}
	err := json.Unmarshal(input, &createFolderInput)
	if err != nil {
		return "", err
	}

	if createFolderInput.Path == "" {
		return "", fmt.Errorf("path cannot be empty")
	}

	err = os.MkdirAll(createFolderInput.Path, 0755)
	if err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	return fmt.Sprintf("Successfully created directory %s", createFolderInput.Path), nil
}

var DeleteFolderDefinition = ToolDefinition{
	Name:        "delete_folder",
	Description: "Delete a directory/folder and all its contents. Use with extreme caution as this action cannot be undone.",
	InputSchema: DeleteFolderInputSchema,
	Function:    DeleteFolder,
}

type DeleteFolderInput struct {
	Path string `json:"path" jsonschema_description:"The path of the directory to delete"`
}

var DeleteFolderInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "The path of the directory to delete",
		},
	},
	"required": []string{"path"},
}

func DeleteFolder(input json.RawMessage) (string, error) {
	deleteFolderInput := DeleteFolderInput{}
	err := json.Unmarshal(input, &deleteFolderInput)
	if err != nil {
		return "", err
	}

	if deleteFolderInput.Path == "" {
		return "", fmt.Errorf("path cannot be empty")
	}

	// Check if directory exists
	if _, err := os.Stat(deleteFolderInput.Path); os.IsNotExist(err) {
		return "", fmt.Errorf("directory does not exist: %s", deleteFolderInput.Path)
	}

	err = os.RemoveAll(deleteFolderInput.Path)
	if err != nil {
		return "", fmt.Errorf("failed to delete directory: %w", err)
	}

	return fmt.Sprintf("Successfully deleted directory %s", deleteFolderInput.Path), nil
}

var RenameFolderDefinition = ToolDefinition{
	Name:        "rename_folder",
	Description: "Rename or move a directory from one location to another.",
	InputSchema: RenameFolderInputSchema,
	Function:    RenameFolder,
}

type RenameFolderInput struct {
	OldPath string `json:"old_path" jsonschema_description:"The current path of the directory"`
	NewPath string `json:"new_path" jsonschema_description:"The new path for the directory"`
}

var RenameFolderInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"old_path": map[string]any{
			"type":        "string",
			"description": "The current path of the directory",
		},
		"new_path": map[string]any{
			"type":        "string",
			"description": "The new path for the directory",
		},
	},
	"required": []string{"old_path", "new_path"},
}

func RenameFolder(input json.RawMessage) (string, error) {
	renameFolderInput := RenameFolderInput{}
	err := json.Unmarshal(input, &renameFolderInput)
	if err != nil {
		return "", err
	}

	if renameFolderInput.OldPath == "" || renameFolderInput.NewPath == "" {
		return "", fmt.Errorf("both old_path and new_path must be provided")
	}

	// Check if source directory exists
	if _, err := os.Stat(renameFolderInput.OldPath); os.IsNotExist(err) {
		return "", fmt.Errorf("source directory does not exist: %s", renameFolderInput.OldPath)
	}

	// Create parent directory for new path if needed
	parentDir := path.Dir(renameFolderInput.NewPath)
	if parentDir != "." {
		err := os.MkdirAll(parentDir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create parent directory: %w", err)
		}
	}

	err = os.Rename(renameFolderInput.OldPath, renameFolderInput.NewPath)
	if err != nil {
		return "", fmt.Errorf("failed to rename directory: %w", err)
	}

	return fmt.Sprintf("Successfully renamed directory %s to %s", renameFolderInput.OldPath, renameFolderInput.NewPath), nil
}

var TerminalRunDefinition = ToolDefinition{
	Name:        "terminal_run",
	Description: "Execute a terminal/command line command and return its output. Use with caution as this can execute any system command.",
	InputSchema: TerminalRunInputSchema,
	Function:    TerminalRun,
}

type TerminalRunInput struct {
	Command string `json:"command" jsonschema_description:"The command to execute in the terminal"`
	Timeout int    `json:"timeout,omitempty" jsonschema_description:"Timeout in seconds (default: 30)"`
}

var TerminalRunInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"command": map[string]any{
			"type":        "string",
			"description": "The command to execute in the terminal",
		},
		"timeout": map[string]any{
			"type":        "integer",
			"description": "Timeout in seconds (default: 30)",
		},
	},
	"required": []string{"command"},
}

func TerminalRun(input json.RawMessage) (string, error) {
	terminalRunInput := TerminalRunInput{}
	err := json.Unmarshal(input, &terminalRunInput)
	if err != nil {
		return "", err
	}

	if terminalRunInput.Command == "" {
		return "", fmt.Errorf("command cannot be empty")
	}

	// Set default timeout
	timeout := 30
	if terminalRunInput.Timeout > 0 {
		timeout = terminalRunInput.Timeout
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	// Execute command based on OS
	var cmd *exec.Cmd
	if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") {
		cmd = exec.CommandContext(ctx, "powershell", "-Command", terminalRunInput.Command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", terminalRunInput.Command)
	}

	// Capture both stdout and stderr
	output, err := cmd.CombinedOutput()

	result := fmt.Sprintf("Command: %s\n", terminalRunInput.Command)
	result += fmt.Sprintf("Exit Code: %d\n", cmd.ProcessState.ExitCode())
	result += fmt.Sprintf("Output:\n%s", string(output))

	if err != nil {
		result += fmt.Sprintf("\nError: %s", err.Error())
	}

	return result, nil
}

var CreateWebsiteDefinition = ToolDefinition{
	Name:        "create_website",
	Description: "Create a complete website with HTML, CSS, and JavaScript files. This tool creates separate files for better organization and handles large content.",
	InputSchema: CreateWebsiteInputSchema,
	Function:    CreateWebsite,
}

type CreateWebsiteInput struct {
	FolderPath  string `json:"folder_path" jsonschema_description:"The folder where the website files should be created"`
	ProjectName string `json:"project_name" jsonschema_description:"Name of the website/project (used for titles and file names)"`
	Description string `json:"description" jsonschema_description:"Brief description of what the website should be about"`
	Style       string `json:"style,omitempty" jsonschema_description:"Style preference (e.g., 'modern', 'minimal', 'colorful', 'professional')"`
}

var CreateWebsiteInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"folder_path": map[string]any{
			"type":        "string",
			"description": "The folder where the website files should be created",
		},
		"project_name": map[string]any{
			"type":        "string",
			"description": "Name of the website/project (used for titles and file names)",
		},
		"description": map[string]any{
			"type":        "string",
			"description": "Brief description of what the website should be about",
		},
		"style": map[string]any{
			"type":        "string",
			"description": "Style preference (e.g., 'modern', 'minimal', 'colorful', 'professional')",
		},
	},
	"required": []string{"folder_path", "project_name", "description"},
}

func CreateWebsite(input json.RawMessage) (string, error) {
	websiteInput := CreateWebsiteInput{}
	err := json.Unmarshal(input, &websiteInput)
	if err != nil {
		return "", err
	}

	if websiteInput.FolderPath == "" || websiteInput.ProjectName == "" || websiteInput.Description == "" {
		return "", fmt.Errorf("folder_path, project_name, and description are required")
	}

	// Create the folder if it doesn't exist
	err = os.MkdirAll(websiteInput.FolderPath, 0755)
	if err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	// Create HTML file
	htmlContent := generateHTML(websiteInput.ProjectName, websiteInput.Description, websiteInput.Style)
	htmlPath := filepath.Join(websiteInput.FolderPath, "index.html")
	err = os.WriteFile(htmlPath, []byte(htmlContent), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create HTML file: %w", err)
	}

	// Create CSS file
	cssContent := generateCSS(websiteInput.Style)
	cssPath := filepath.Join(websiteInput.FolderPath, "style.css")
	err = os.WriteFile(cssPath, []byte(cssContent), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create CSS file: %w", err)
	}

	// Create JS file
	jsContent := generateJS(websiteInput.ProjectName)
	jsPath := filepath.Join(websiteInput.FolderPath, "script.js")
	err = os.WriteFile(jsPath, []byte(jsContent), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create JS file: %w", err)
	}

	return fmt.Sprintf("Successfully created website '%s' in %s with files: index.html, style.css, script.js", websiteInput.ProjectName, websiteInput.FolderPath), nil
}

func generateHTML(projectName, description, style string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%s</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <nav>
            <div class="logo">%s</div>
            <ul class="nav-links">
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <section id="home" class="hero">
            <div class="hero-content">
                <h1>Welcome to %s</h1>
                <p>%s</p>
                <button class="cta-button" onclick="showAlert()">Get Started</button>
            </div>
        </section>

        <section id="about" class="about">
            <div class="container">
                <h2>About Us</h2>
                <p>We are dedicated to providing excellent service and innovative solutions.</p>
            </div>
        </section>

        <section id="services" class="services">
            <div class="container">
                <h2>Our Services</h2>
                <div class="service-grid">
                    <div class="service-card">
                        <h3>Service 1</h3>
                        <p>High-quality service description here.</p>
                    </div>
                    <div class="service-card">
                        <h3>Service 2</h3>
                        <p>Another excellent service we provide.</p>
                    </div>
                    <div class="service-card">
                        <h3>Service 3</h3>
                        <p>Premium service with great value.</p>
                    </div>
                </div>
            </div>
        </section>

        <section id="contact" class="contact">
            <div class="container">
                <h2>Contact Us</h2>
                <p>Get in touch with us today!</p>
                <button class="contact-button" onclick="showContact()">Contact Now</button>
            </div>
        </section>
    </main>

    <footer>
        <p>&copy; 2025 %s. All rights reserved.</p>
    </footer>

    <script src="script.js"></script>
</body>
</html>`, projectName, projectName, projectName, description, projectName)
}

func generateCSS(style string) string {
	return `* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

header {
    background: #fff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    position: fixed;
    width: 100%;
    top: 0;
    z-index: 1000;
}

nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2c3e50;
}

.nav-links {
    display: flex;
    list-style: none;
    gap: 2rem;
}

.nav-links a {
    text-decoration: none;
    color: #333;
    transition: color 0.3s;
}

.nav-links a:hover {
    color: #3498db;
}

main {
    margin-top: 80px;
}

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 100px 0;
    text-align: center;
}

.hero-content h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.hero-content p {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.cta-button, .contact-button {
    background: #e74c3c;
    color: white;
    border: none;
    padding: 15px 30px;
    font-size: 1.1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.3s;
}

.cta-button:hover, .contact-button:hover {
    background: #c0392b;
}

.about, .services, .contact {
    padding: 80px 0;
}

.about {
    background: #f8f9fa;
}

.about h2, .services h2, .contact h2 {
    text-align: center;
    margin-bottom: 3rem;
    font-size: 2.5rem;
    color: #2c3e50;
}

.service-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 3rem;
}

.service-card {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.3s;
}

.service-card:hover {
    transform: translateY(-5px);
}

.service-card h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.contact {
    background: #2c3e50;
    color: white;
    text-align: center;
}

.contact h2 {
    color: white;
}

footer {
    background: #34495e;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

@media (max-width: 768px) {
    .nav-links {
        display: none;
    }

    .hero-content h1 {
        font-size: 2rem;
    }

    .service-grid {
        grid-template-columns: 1fr;
    }
}`
}

func generateJS(projectName string) string {
	return fmt.Sprintf(`// %s Website JavaScript

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);

            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// CTA Button functionality
function showAlert() {
    alert('Welcome to %s! This is a demo website created by the AI agent.');
}

// Contact button functionality
function showContact() {
    const email = 'contact@%s.com';
    const phone = '+1 (555) 123-4567';

    alert('Contact Information:\\n\\nEmail: ' + email + '\\nPhone: ' + phone + '\\n\\nThis is a demo contact - replace with real information!');
}

// Add some interactive animations
document.addEventListener('DOMContentLoaded', function() {
    // Animate service cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// Add mobile menu toggle (basic implementation)
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('mobile-active');
}

// Console welcome message
console.log('Welcome to %s! This website was created by an AI agent.');
console.log('Feel free to explore the code and customize it to your needs!');`, projectName, projectName, projectName, projectName)
}
