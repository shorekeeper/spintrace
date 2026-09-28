//! Win32 ABI declarations.
//!
//! Only declarations used by the window, input translator and Vulkan loader
//! are retained. Handles remain opaque pointers and structures follow their C
//! field order so they can cross the system ABI without conversion.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type BOOL = i32;
pub type UINT = u32;
pub type DWORD = u32;
pub type LPARAM = isize;
pub type WPARAM = usize;
pub type LRESULT = isize;
pub type LONG_PTR = isize;
pub type ATOM = u16;
pub type HRESULT = i32;
pub type DPI_AWARENESS_CONTEXT = isize;

pub type HANDLE = *mut c_void;
pub type HMODULE = HANDLE;
pub type HINSTANCE = HANDLE;
pub type HWND = HANDLE;
pub type HICON = HANDLE;
pub type HCURSOR = HANDLE;
pub type HBRUSH = HANDLE;
pub type HMENU = HANDLE;
pub type HMONITOR = HANDLE;

pub type WNDPROC =
    Option<unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT>;

pub const FALSE: BOOL = 0;
pub const TRUE: BOOL = 1;

pub const CS_VREDRAW: UINT = 0x0001;
pub const CS_HREDRAW: UINT = 0x0002;
pub const CS_DBLCLKS: UINT = 0x0008;

pub const WS_POPUP: DWORD = 0x8000_0000;
pub const WS_THICKFRAME: DWORD = 0x0004_0000;
pub const WS_MINIMIZEBOX: DWORD = 0x0002_0000;
pub const WS_MAXIMIZEBOX: DWORD = 0x0001_0000;
pub const WS_SYSMENU: DWORD = 0x0008_0000;
pub const WS_SPINTRACE_WINDOW: DWORD =
    WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU;

pub const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
pub const SW_SHOW: i32 = 5;
pub const SW_MINIMIZE: i32 = 6;
pub const SW_MAXIMIZE: i32 = 3;
pub const SW_RESTORE: i32 = 9;

pub const SIZE_RESTORED: WPARAM = 0;
pub const SIZE_MINIMIZED: WPARAM = 1;
pub const SIZE_MAXIMIZED: WPARAM = 2;

pub const PM_REMOVE: UINT = 0x0001;
pub const GWLP_USERDATA: i32 = -21;

pub const SWP_NOSIZE: UINT = 0x0001;
pub const SWP_NOMOVE: UINT = 0x0002;
pub const SWP_NOZORDER: UINT = 0x0004;
pub const SWP_NOACTIVATE: UINT = 0x0010;
pub const SWP_FRAMECHANGED: UINT = 0x0020;

pub const HTCLIENT: LRESULT = 1;
pub const HTCAPTION: LRESULT = 2;
pub const HTLEFT: LRESULT = 10;
pub const HTRIGHT: LRESULT = 11;
pub const HTTOP: LRESULT = 12;
pub const HTTOPLEFT: LRESULT = 13;
pub const HTTOPRIGHT: LRESULT = 14;
pub const HTBOTTOM: LRESULT = 15;
pub const HTBOTTOMLEFT: LRESULT = 16;
pub const HTBOTTOMRIGHT: LRESULT = 17;

pub const MONITOR_DEFAULTTONEAREST: DWORD = 2;

pub const WM_DESTROY: UINT = 0x0002;
pub const WM_SIZE: UINT = 0x0005;
pub const WM_SETFOCUS: UINT = 0x0007;
pub const WM_KILLFOCUS: UINT = 0x0008;
pub const WM_CLOSE: UINT = 0x0010;
pub const WM_QUIT: UINT = 0x0012;
pub const WM_ERASEBKGND: UINT = 0x0014;
pub const WM_SETCURSOR: UINT = 0x0020;
pub const WM_GETMINMAXINFO: UINT = 0x0024;
pub const WM_KEYDOWN: UINT = 0x0100;
pub const WM_KEYUP: UINT = 0x0101;
pub const WM_CHAR: UINT = 0x0102;
pub const WM_SYSKEYDOWN: UINT = 0x0104;
pub const WM_SYSKEYUP: UINT = 0x0105;
pub const WM_NCCREATE: UINT = 0x0081;
pub const WM_NCDESTROY: UINT = 0x0082;
pub const WM_NCCALCSIZE: UINT = 0x0083;
pub const WM_NCHITTEST: UINT = 0x0084;
pub const WM_NCLBUTTONDOWN: UINT = 0x00A1;
pub const WM_MOUSEMOVE: UINT = 0x0200;
pub const WM_LBUTTONDOWN: UINT = 0x0201;
pub const WM_LBUTTONUP: UINT = 0x0202;
pub const WM_LBUTTONDBLCLK: UINT = 0x0203;
pub const WM_RBUTTONDOWN: UINT = 0x0204;
pub const WM_RBUTTONUP: UINT = 0x0205;
pub const WM_RBUTTONDBLCLK: UINT = 0x0206;
pub const WM_MBUTTONDOWN: UINT = 0x0207;
pub const WM_MBUTTONUP: UINT = 0x0208;
pub const WM_MBUTTONDBLCLK: UINT = 0x0209;
pub const WM_MOUSEWHEEL: UINT = 0x020A;
pub const WM_XBUTTONDOWN: UINT = 0x020B;
pub const WM_XBUTTONUP: UINT = 0x020C;
pub const WM_XBUTTONDBLCLK: UINT = 0x020D;
pub const WM_MOUSEHWHEEL: UINT = 0x020E;
pub const WM_MOUSELEAVE: UINT = 0x02A3;

pub const VK_BACK: i32 = 0x08;
pub const VK_TAB: i32 = 0x09;
pub const VK_RETURN: i32 = 0x0D;
pub const VK_SHIFT: i32 = 0x10;
pub const VK_CONTROL: i32 = 0x11;
pub const VK_MENU: i32 = 0x12;
pub const VK_ESCAPE: i32 = 0x1B;
pub const VK_SPACE: i32 = 0x20;
pub const VK_PRIOR: i32 = 0x21;
pub const VK_NEXT: i32 = 0x22;
pub const VK_END: i32 = 0x23;
pub const VK_HOME: i32 = 0x24;
pub const VK_LEFT: i32 = 0x25;
pub const VK_UP: i32 = 0x26;
pub const VK_RIGHT: i32 = 0x27;
pub const VK_DOWN: i32 = 0x28;
pub const VK_DELETE: i32 = 0x2E;

pub const XBUTTON1: u16 = 1;
pub const XBUTTON2: u16 = 2;
pub const WHEEL_DELTA: f32 = 120.0;

pub const TME_LEAVE: DWORD = 0x0000_0002;
pub const HOVER_DEFAULT: DWORD = 0xFFFF_FFFF;

pub const IDC_ARROW: usize = 32512;
pub const IDC_IBEAM: usize = 32513;
pub const IDC_SIZEWE: usize = 32644;
pub const IDC_SIZENS: usize = 32645;
pub const IDC_HAND: usize = 32649;

pub const DWMWA_USE_IMMERSIVE_DARK_MODE: DWORD = 20;
pub const DWMWA_WINDOW_CORNER_PREFERENCE: DWORD = 33;
pub const DWMWA_BORDER_COLOR: DWORD = 34;
pub const DWMWCP_DONOTROUND: DWORD = 1;
pub const DWMWA_COLOR_NONE: DWORD = 0xFFFF_FFFE;

pub const WM_DPICHANGED: UINT = 0x02E0;
pub const USER_DEFAULT_SCREEN_DPI: u32 = 96;
pub const DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: DPI_AWARENESS_CONTEXT = -4;

pub const ERROR_CLASS_ALREADY_EXISTS: DWORD = 1410;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct POINT {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RECT {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
pub struct MSG {
    pub hwnd: HWND,
    pub message: UINT,
    pub wParam: WPARAM,
    pub lParam: LPARAM,
    pub time: DWORD,
    pub pt: POINT,
    pub lPrivate: DWORD,
}

#[repr(C)]
pub struct WNDCLASSEXW {
    pub cbSize: UINT,
    pub style: UINT,
    pub lpfnWndProc: WNDPROC,
    pub cbClsExtra: i32,
    pub cbWndExtra: i32,
    pub hInstance: HINSTANCE,
    pub hIcon: HICON,
    pub hCursor: HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName: *const u16,
    pub lpszClassName: *const u16,
    pub hIconSm: HICON,
}

#[repr(C)]
pub struct CREATESTRUCTW {
    pub lpCreateParams: *mut c_void,
    pub hInstance: HINSTANCE,
    pub hMenu: HMENU,
    pub hwndParent: HWND,
    pub cy: i32,
    pub cx: i32,
    pub y: i32,
    pub x: i32,
    pub style: i32,
    pub lpszName: *const u16,
    pub lpszClass: *const u16,
    pub dwExStyle: DWORD,
}

#[repr(C)]
pub struct TRACKMOUSEEVENT {
    pub cbSize: DWORD,
    pub dwFlags: DWORD,
    pub hwndTrack: HWND,
    pub dwHoverTime: DWORD,
}

#[repr(C)]
pub struct MINMAXINFO {
    pub ptReserved: POINT,
    pub ptMaxSize: POINT,
    pub ptMaxPosition: POINT,
    pub ptMinTrackSize: POINT,
    pub ptMaxTrackSize: POINT,
}

#[repr(C)]
pub struct MONITORINFO {
    pub cbSize: DWORD,
    pub rcMonitor: RECT,
    pub rcWork: RECT,
    pub dwFlags: DWORD,
}

#[link(name = "kernel32")]
extern "system" {
    pub fn LoadLibraryW(name: *const u16) -> HMODULE;
    pub fn GetProcAddress(module: HMODULE, name: *const i8) -> *const c_void;
    pub fn GetModuleHandleW(name: *const u16) -> HMODULE;
    pub fn GetWindowsDirectoryW(buffer: *mut u16, size: UINT) -> UINT;
    pub fn GetLastError() -> DWORD;
}

#[link(name = "user32")]
extern "system" {
    pub fn RegisterClassExW(class: *const WNDCLASSEXW) -> ATOM;
    pub fn CreateWindowExW(
        ex_style: DWORD,
        class_name: *const u16,
        window_name: *const u16,
        style: DWORD,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: HWND,
        menu: HMENU,
        instance: HINSTANCE,
        param: *mut c_void,
    ) -> HWND;
    pub fn DefWindowProcW(hwnd: HWND, message: UINT, wparam: WPARAM, lparam: LPARAM) -> LRESULT;
    pub fn DestroyWindow(hwnd: HWND) -> BOOL;
    pub fn ShowWindow(hwnd: HWND, command: i32) -> BOOL;
    pub fn UpdateWindow(hwnd: HWND) -> BOOL;
    pub fn AdjustWindowRectEx(rect: *mut RECT, style: DWORD, menu: BOOL, ex_style: DWORD) -> BOOL;
    pub fn GetClientRect(hwnd: HWND, rect: *mut RECT) -> BOOL;
    pub fn GetWindowRect(hwnd: HWND, rect: *mut RECT) -> BOOL;
    pub fn PeekMessageW(
        message: *mut MSG,
        hwnd: HWND,
        min_filter: UINT,
        max_filter: UINT,
        remove: UINT,
    ) -> BOOL;
    pub fn TranslateMessage(message: *const MSG) -> BOOL;
    pub fn DispatchMessageW(message: *const MSG) -> LRESULT;
    pub fn PostQuitMessage(exit_code: i32);
    pub fn PostMessageW(hwnd: HWND, message: UINT, wparam: WPARAM, lparam: LPARAM) -> BOOL;
    pub fn SendMessageW(hwnd: HWND, message: UINT, wparam: WPARAM, lparam: LPARAM) -> LRESULT;
    pub fn LoadCursorW(instance: HINSTANCE, name: *const u16) -> HCURSOR;
    pub fn SetCursor(cursor: HCURSOR) -> HCURSOR;
    pub fn SetCapture(hwnd: HWND) -> HWND;
    pub fn ReleaseCapture() -> BOOL;
    pub fn TrackMouseEvent(event: *mut TRACKMOUSEEVENT) -> BOOL;
    pub fn ScreenToClient(hwnd: HWND, point: *mut POINT) -> BOOL;
    pub fn GetKeyState(key: i32) -> i16;
    pub fn SetWindowLongPtrW(hwnd: HWND, index: i32, value: LONG_PTR) -> LONG_PTR;
    pub fn GetWindowLongPtrW(hwnd: HWND, index: i32) -> LONG_PTR;
    pub fn SetWindowPos(
        hwnd: HWND,
        insert_after: HWND,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        flags: UINT,
    ) -> BOOL;
    pub fn IsZoomed(hwnd: HWND) -> BOOL;
    pub fn SetProcessDpiAwarenessContext(value: DPI_AWARENESS_CONTEXT) -> BOOL;
    pub fn GetDpiForWindow(hwnd: HWND) -> UINT;
    pub fn MonitorFromWindow(hwnd: HWND, flags: DWORD) -> HMONITOR;
    pub fn GetMonitorInfoW(monitor: HMONITOR, info: *mut MONITORINFO) -> BOOL;
}

#[link(name = "dwmapi")]
extern "system" {
    pub fn DwmSetWindowAttribute(
        hwnd: HWND,
        attribute: DWORD,
        value: *const c_void,
        size: DWORD,
    ) -> HRESULT;
}