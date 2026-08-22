# Send keys to an Isaac window without focusing it, then capture the result.
#
#   .\keysend.ps1 -Keys ENTER,DOWN,ENTER -Shot step1
#
# Keys are posted with WM_KEYDOWN/WM_KEYUP, which Isaac's menus accept from a
# background window. This is what lets many instances be set up in parallel.

param(
    [string[]]$Keys = @(),
    [string]$Shot = "shot",
    [int]$Index = 0,
    [int]$DelayMs = 350
)

Add-Type @"
using System; using System.Runtime.InteropServices; using System.Drawing;
public class IsaacWin {
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint f);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  public struct RECT { public int L, T, R, B; }
  public static void Key(IntPtr h, int vk, int holdMs) {
    PostMessage(h, 0x100, (IntPtr)vk, (IntPtr)0);
    System.Threading.Thread.Sleep(holdMs);
    PostMessage(h, 0x101, (IntPtr)vk, (IntPtr)0);
  }
  public static Bitmap Grab(IntPtr h) {
    RECT r; GetClientRect(h, out r);
    if (r.R - r.L <= 0) return null;
    Bitmap b = new Bitmap(r.R - r.L, r.B - r.T);
    using (Graphics g = Graphics.FromImage(b)) {
      IntPtr hdc = g.GetHdc(); PrintWindow(h, hdc, 3); g.ReleaseHdc(hdc);
    }
    return b;
  }
}
"@ -ReferencedAssemblies System.Drawing

$VK = @{
    "ENTER" = 0x0D; "SPACE" = 0x20; "ESC" = 0x1B; "TAB" = 0x09;
    "UP" = 0x26; "DOWN" = 0x28; "LEFT" = 0x25; "RIGHT" = 0x27;
    "Z" = 0x5A; "X" = 0x58; "R" = 0x52; "W" = 0x57; "A" = 0x41;
    "S" = 0x53; "D" = 0x44; "BACKTICK" = 0xC0
}

$procs = @(Get-Process -Name "isaac-ng" -ErrorAction SilentlyContinue | Sort-Object Id)
if ($procs.Count -eq 0) { Write-Output "no isaac instance"; exit 1 }
$h = $procs[$Index].MainWindowHandle

foreach ($k in $Keys) {
    $code = $VK[$k.ToUpper()]
    if ($null -eq $code) { Write-Output "unknown key: $k"; continue }
    [IsaacWin]::Key($h, $code, 120)
    Start-Sleep -Milliseconds $DelayMs
}

Start-Sleep -Milliseconds 600
$bmp = [IsaacWin]::Grab($h)
$out = "C:\Users\maxpa\OneDrive\Documents\Isaac AI Claude\spike\$Shot.png"
$bmp.Save($out)
$bmp.Dispose()
Write-Output "sent: $($Keys -join ',') -> $out"
