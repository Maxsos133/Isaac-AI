# Focus an Isaac window reliably and send real keystrokes to it.
#
# Windows refuses SetForegroundWindow from a background process, so we attach to
# the current foreground thread's input queue first. This is only needed for
# menu navigation at startup; gameplay input goes through the mod and never
# needs focus.

param(
    [string[]]$Keys = @(),
    [string]$Shot = "focused",
    [int]$Index = 0,
    [int]$DelayMs = 400
)

Add-Type @"
using System; using System.Runtime.InteropServices; using System.Drawing;
public class IsaacFocus {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr pid);
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
  [DllImport("user32.dll")] public static extern uint SendInput(uint n, INPUT[] pInputs, int cb);
  [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint mapType);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint f);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);

  public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)]
  public struct INPUT { public uint type; public KEYBDINPUT ki; public int pad1; public int pad2; }
  [StructLayout(LayoutKind.Sequential)]
  public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr extra; }

  public static bool Focus(IntPtr h) {
    ShowWindow(h, 9); // SW_RESTORE
    uint fg = GetWindowThreadProcessId(GetForegroundWindow(), IntPtr.Zero);
    uint me = GetCurrentThreadId();
    AttachThreadInput(me, fg, true);
    BringWindowToTop(h);
    bool ok = SetForegroundWindow(h);
    AttachThreadInput(me, fg, false);
    System.Threading.Thread.Sleep(250);
    return GetForegroundWindow() == h;
  }

  public static void Key(ushort vk, int holdMs) {
    ushort scan = (ushort)MapVirtualKey(vk, 0);
    INPUT[] down = new INPUT[1];
    down[0].type = 1;
    down[0].ki.wVk = vk; down[0].ki.wScan = scan; down[0].ki.dwFlags = 0x0008; // SCANCODE
    SendInput(1, down, Marshal.SizeOf(typeof(INPUT)));
    System.Threading.Thread.Sleep(holdMs);
    INPUT[] up = new INPUT[1];
    up[0].type = 1;
    up[0].ki.wVk = vk; up[0].ki.wScan = scan; up[0].ki.dwFlags = 0x0008 | 0x0002; // SCANCODE|KEYUP
    SendInput(1, up, Marshal.SizeOf(typeof(INPUT)));
  }

  public static void Shot(IntPtr h, string path) {
    RECT r; GetClientRect(h, out r);
    using (Bitmap b = new Bitmap(r.R-r.L, r.B-r.T)) {
      using (Graphics g = Graphics.FromImage(b)) { IntPtr hdc = g.GetHdc(); PrintWindow(h, hdc, 3); g.ReleaseHdc(hdc); }
      b.Save(path);
    }
  }
}
"@ -ReferencedAssemblies System.Drawing

$VK = @{
    "ENTER"=0x0D; "SPACE"=0x20; "ESC"=0x1B; "TAB"=0x09;
    # Arrow keys are ATTACK in Isaac. Menu navigation uses the MOVE keys: WASD.
    "UP"=0x26; "DOWN"=0x28; "LEFT"=0x25; "RIGHT"=0x27;
    "W"=0x57; "A"=0x41; "S"=0x53; "D"=0x44;
    "Z"=0x5A; "X"=0x58; "R"=0x52; "E"=0x45; "Q"=0x51; "BACKTICK"=0xC0
}

$procs = @(Get-Process -Name "isaac-ng" -ErrorAction SilentlyContinue | Sort-Object Id)
if ($procs.Count -eq 0) { Write-Output "no isaac instance"; exit 1 }
$h = $procs[$Index].MainWindowHandle

$focused = [IsaacFocus]::Focus($h)
Write-Output "focused=$focused"

foreach ($k in $Keys) {
    $code = $VK[$k.ToUpper()]
    if ($null -eq $code) { Write-Output "unknown key: $k"; continue }
    [IsaacFocus]::Key([uint16]$code, 90)
    Start-Sleep -Milliseconds $DelayMs
}

Start-Sleep -Milliseconds 700
[IsaacFocus]::Shot($h, "C:\Users\maxpa\OneDrive\Documents\Isaac AI Claude\spike\$Shot.png")
Write-Output "sent: $($Keys -join ',')"
