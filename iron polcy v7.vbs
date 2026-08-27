Option Explicit

Dim shell, fileSystem, projectFolder, launcher
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)
launcher = fileSystem.BuildPath(projectFolder, "BASLAT.bat")
shell.CurrentDirectory = projectFolder
shell.Run Chr(34) & launcher & Chr(34), 0, False
