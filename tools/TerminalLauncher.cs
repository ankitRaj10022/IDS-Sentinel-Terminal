using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;

public static class TerminalLauncher
{
    public static int Main(string[] args)
    {
        string rootDir = AppDomain.CurrentDomain.BaseDirectory;
        string moduleArgs = "-X utf8 -m ids_app.product_terminal" + BuildUserArgs(args);

        try
        {
            return RunPython("py", "-3 " + moduleArgs, rootDir);
        }
        catch (Win32Exception)
        {
            try
            {
                return RunPython("python", moduleArgs, rootDir);
            }
            catch (Win32Exception)
            {
                Console.Error.WriteLine("Python was not found. Install Python 3, then run terminal.exe again.");
                return 1;
            }
        }
    }

    private static int RunPython(string executable, string arguments, string rootDir)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = executable;
        startInfo.Arguments = arguments;
        startInfo.WorkingDirectory = rootDir;
        startInfo.UseShellExecute = false;
        startInfo.EnvironmentVariables["PYTHONUTF8"] = "1";

        using (Process process = Process.Start(startInfo))
        {
            process.WaitForExit();
            return process.ExitCode;
        }
    }

    private static string BuildUserArgs(string[] args)
    {
        if (args == null || args.Length == 0)
        {
            return "";
        }

        StringBuilder builder = new StringBuilder();
        foreach (string arg in args)
        {
            builder.Append(" ");
            builder.Append(Quote(arg));
        }
        return builder.ToString();
    }

    private static string Quote(string value)
    {
        if (String.IsNullOrEmpty(value))
        {
            return "\"\"";
        }
        if (value.IndexOfAny(new char[] { ' ', '\t', '\n', '\r', '"' }) < 0)
        {
            return value;
        }
        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
}
