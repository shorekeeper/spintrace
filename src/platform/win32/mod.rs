//! Win32 window ownership and message translation.
//!
//! WindowState is allocated before CreateWindowExW and its stable address is
//! passed through CREATESTRUCTW. The window procedure stores that address in
//! GWLP_USERDATA, so native callbacks append events directly to the owning
//! window without global mutable state.
//!
//! The ordinary non-client frame is removed by WM_NCCALCSIZE. Resize hit tests
//! remain native, which preserves edge dragging and snap behaviour while the
//! caption, border and window buttons are drawn by the application.
//!
//! Mouse capture begins with a button press and ends after the final release.
//! This keeps a drag alive outside the client rectangle and guarantees that its
//! release is delivered to the window that started it.

pub mod ffi;

use std::ffi::c_void;
use std::path::PathBuf;

use crate::core::error::Category;
use crate::core::{Error, Result};
use crate::platform::{CursorKind, Event, Key, Modifiers, MouseButton};

use ffi::*;

const BUTTON_LEFT: u8 = 1 << 0;
const BUTTON_RIGHT: u8 = 1 << 1;
const BUTTON_MIDDLE: u8 = 1 << 2;
const BUTTON_X1: u8 = 1 << 3;
const BUTTON_X2: u8 = 1 << 4;
const RESIZE_BORDER_PX: i32 = 7;

struct WindowState {
    width: u32,
    height: u32,
    normal_width: u32,
    normal_height: u32,
    maximized: bool,
    dpi: u32,
    closed: bool,
    tracking_mouse: bool,
    buttons: u8,
    high_surrogate: Option<u16>,
    cursor: CursorKind,
    events: Vec<Event>,
}

/// Owned native window and its translated event queue.
pub struct Window {
    hwnd: HWND,
    hinstance: HINSTANCE,
    state: Box<WindowState>,
}

impl Window {
    pub fn new(title: &str, width: u32, height: u32) -> Result<Window> {
        unsafe {
            // The call must precede every DPI dependent API and window creation.
            // Failure is harmless when a manifest or an earlier call already
            // established the process awareness.
            SetProcessDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
            );

            let hinstance = GetModuleHandleW(std::ptr::null());
            if hinstance.is_null() {
                return Err(last_error("GetModuleHandleW"));
            }

            let class_name = wide("spintrace.window");
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
                lpfnWndProc: Some(window_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: std::ptr::null_mut(),
                hCursor: cursor_handle(CursorKind::Arrow),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class_name.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };

            if RegisterClassExW(&class) == 0 {
                let code = GetLastError();
                if code != ERROR_CLASS_ALREADY_EXISTS {
                    return Err(Error::with_code(
                        Category::Platform,
                        "RegisterClassExW failed",
                        code as i64,
                    ));
                }
            }

            let style = WS_SPINTRACE_WINDOW;
            let mut outer =
                RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
            if AdjustWindowRectEx(&mut outer, style, FALSE, 0) == 0 {
                return Err(last_error("AdjustWindowRectEx"));
            }

            let mut state = Box::new(WindowState {
                width,
                height,
                normal_width: width,
                normal_height: height,
                maximized: false,
                dpi: USER_DEFAULT_SCREEN_DPI,
                closed: false,
                tracking_mouse: false,
                buttons: 0,
                high_surrogate: None,
                cursor: CursorKind::Arrow,
                events: Vec::with_capacity(64),
            });

            let title = wide(title);
            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                title.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                outer.right - outer.left,
                outer.bottom - outer.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                state.as_mut() as *mut WindowState as *mut c_void,
            );
            if hwnd.is_null() {
                return Err(last_error("CreateWindowExW"));
            }

            let dpi = GetDpiForWindow(hwnd);
            if dpi > 0 {
                state.dpi = dpi;
            }

            // Unsupported DWM attributes are ignored. WM_NCCALCSIZE remains the
            // authoritative frame removal path on systems that predate them.
            let dark = TRUE;
            let square = DWMWCP_DONOTROUND;
            let no_border = DWMWA_COLOR_NONE;
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &dark as *const _ as *const c_void,
                std::mem::size_of_val(&dark) as u32,
            );
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                &square as *const _ as *const c_void,
                std::mem::size_of_val(&square) as u32,
            );
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                &no_border as *const _ as *const c_void,
                std::mem::size_of_val(&no_border) as u32,
            );

            // Forces the first non-client calculation before the window becomes
            // visible, avoiding one frame with the system caption.
            SetWindowPos(
                hwnd,
                std::ptr::null_mut(),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            );

            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);
            Ok(Window { hwnd, hinstance, state })
        }
    }

    /// Pumps all queued native messages and returns their translated events.
    ///
    /// The output vector is reused by the caller. Events produced while
    /// DispatchMessageW runs are drained only after the native queue is empty,
    /// preserving their original order.
    pub fn poll_events(&mut self, output: &mut Vec<Event>) -> bool {
        output.clear();
        if self.state.closed {
            return false;
        }

        unsafe {
            let mut message: MSG = std::mem::zeroed();
            while PeekMessageW(&mut message, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                if message.message == WM_QUIT {
                    self.state.closed = true;
                    break;
                }
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }

        output.extend(self.state.events.drain(..));
        !self.state.closed
    }

    pub fn client_size(&self) -> (u32, u32) {
        unsafe {
            let mut rect = RECT::default();
            if GetClientRect(self.hwnd, &mut rect) != 0 {
                return (
                    (rect.right - rect.left).max(0) as u32,
                    (rect.bottom - rect.top).max(0) as u32,
                );
            }
        }
        (self.state.width, self.state.height)
    }

    pub fn set_cursor(&mut self, cursor: CursorKind) {
        self.state.cursor = cursor;
        unsafe {
            SetCursor(cursor_handle(cursor));
        }
    }

    /// Hands the current pointer gesture to the system move loop.
    ///
    /// The GUI clears its held state before calling this because the modal move
    /// loop consumes the release message.
    pub fn begin_drag(&mut self) {
        unsafe {
            ReleaseCapture();
            SendMessageW(self.hwnd, WM_NCLBUTTONDOWN, HTCAPTION as usize, 0);
        }
    }

    pub fn minimize(&mut self) {
        unsafe {
            ShowWindow(self.hwnd, SW_MINIMIZE);
        }
    }

    pub fn toggle_maximized(&mut self) {
        unsafe {
            ShowWindow(
                self.hwnd,
                if IsZoomed(self.hwnd) != 0 { SW_RESTORE } else { SW_MAXIMIZE },
            );
        }
    }

    pub fn is_maximized(&self) -> bool {
        self.state.maximized
    }

    /// Last client size reported while the window was restored.
    pub fn normal_size(&self) -> (u32, u32) {
        (self.state.normal_width, self.state.normal_height)
    }

    pub fn dpi_scale(&self) -> f32 {
        self.state.dpi.max(USER_DEFAULT_SCREEN_DPI) as f32
            / USER_DEFAULT_SCREEN_DPI as f32
    }

    pub fn close(&mut self) {
        unsafe {
            PostMessageW(self.hwnd, WM_CLOSE, 0, 0);
        }
    }

    pub fn hwnd(&self) -> *mut c_void {
        self.hwnd
    }

    pub fn hinstance(&self) -> *mut c_void {
        self.hinstance
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        if !self.hwnd.is_null() && !self.state.closed {
            unsafe {
                DestroyWindow(self.hwnd);
            }
        }
        self.hwnd = std::ptr::null_mut();
    }
}

pub fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn windows_directory() -> Option<PathBuf> {
    let mut buffer = vec![0u16; 32_768];
    let length = unsafe { GetWindowsDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
    if length == 0 || length as usize >= buffer.len() {
        return None;
    }
    buffer.truncate(length as usize);
    Some(PathBuf::from(String::from_utf16_lossy(&buffer)))
}

fn last_error(operation: &str) -> Error {
    let code = unsafe { GetLastError() };
    Error::with_code(Category::Platform, format!("{} failed", operation), code as i64)
}

fn cursor_handle(cursor: CursorKind) -> HCURSOR {
    let id = match cursor {
        CursorKind::Arrow => IDC_ARROW,
        CursorKind::Hand => IDC_HAND,
        CursorKind::Text => IDC_IBEAM,
        CursorKind::ResizeHorizontal => IDC_SIZEWE,
        CursorKind::ResizeVertical => IDC_SIZENS,
    };
    unsafe { LoadCursorW(std::ptr::null_mut(), id as *const u16) }
}

fn modifiers() -> Modifiers {
    let down = |key| unsafe { GetKeyState(key) as u16 & 0x8000 != 0 };
    Modifiers {
        shift: down(VK_SHIFT),
        ctrl: down(VK_CONTROL),
        alt: down(VK_MENU),
    }
}

fn key_from_virtual(value: u32) -> Key {
    match value as i32 {
        VK_TAB => Key::Tab,
        VK_BACK => Key::Backspace,
        VK_DELETE => Key::Delete,
        VK_LEFT => Key::Left,
        VK_RIGHT => Key::Right,
        VK_UP => Key::Up,
        VK_DOWN => Key::Down,
        VK_HOME => Key::Home,
        VK_END => Key::End,
        VK_PRIOR => Key::PageUp,
        VK_NEXT => Key::PageDown,
        VK_RETURN => Key::Enter,
        VK_ESCAPE => Key::Escape,
        VK_SPACE => Key::Space,
        _ => Key::Unknown(value),
    }
}

fn message_point(lparam: LPARAM) -> POINT {
    let bits = lparam as u32;
    POINT {
        x: bits as u16 as i16 as i32,
        y: (bits >> 16) as u16 as i16 as i32,
    }
}

fn client_point(lparam: LPARAM) -> (f32, f32) {
    let point = message_point(lparam);
    (point.x as f32, point.y as f32)
}

fn screen_point(hwnd: HWND, lparam: LPARAM) -> (f32, f32) {
    let mut point = message_point(lparam);
    unsafe {
        ScreenToClient(hwnd, &mut point);
    }
    (point.x as f32, point.y as f32)
}

fn high_word(value: usize) -> u16 {
    ((value >> 16) & 0xFFFF) as u16
}

fn button_mask(button: MouseButton) -> u8 {
    match button {
        MouseButton::Left => BUTTON_LEFT,
        MouseButton::Right => BUTTON_RIGHT,
        MouseButton::Middle => BUTTON_MIDDLE,
        MouseButton::X1 => BUTTON_X1,
        MouseButton::X2 => BUTTON_X2,
    }
}

unsafe fn resize_hit(hwnd: HWND, lparam: LPARAM) -> LRESULT {
    if IsZoomed(hwnd) != 0 {
        return HTCLIENT;
    }

    let point = message_point(lparam);
    let mut rect = RECT::default();
    if GetWindowRect(hwnd, &mut rect) == 0 {
        return HTCLIENT;
    }

    let left = point.x < rect.left + RESIZE_BORDER_PX;
    let right = point.x >= rect.right - RESIZE_BORDER_PX;
    let top = point.y < rect.top + RESIZE_BORDER_PX;
    let bottom = point.y >= rect.bottom - RESIZE_BORDER_PX;

    match (left, right, top, bottom) {
        (true, _, true, _) => HTTOPLEFT,
        (_, true, true, _) => HTTOPRIGHT,
        (true, _, _, true) => HTBOTTOMLEFT,
        (_, true, _, true) => HTBOTTOMRIGHT,
        (true, _, _, _) => HTLEFT,
        (_, true, _, _) => HTRIGHT,
        (_, _, true, _) => HTTOP,
        (_, _, _, true) => HTBOTTOM,
        _ => HTCLIENT,
    }
}

unsafe fn constrain_maximized(hwnd: HWND, lparam: LPARAM) {
    let target = lparam as *mut MINMAXINFO;
    if target.is_null() {
        return;
    }

    let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    if monitor.is_null() {
        return;
    }

    let mut info = MONITORINFO {
        cbSize: std::mem::size_of::<MONITORINFO>() as u32,
        rcMonitor: RECT::default(),
        rcWork: RECT::default(),
        dwFlags: 0,
    };
    if GetMonitorInfoW(monitor, &mut info) == 0 {
        return;
    }

    (*target).ptMaxPosition.x = info.rcWork.left - info.rcMonitor.left;
    (*target).ptMaxPosition.y = info.rcWork.top - info.rcMonitor.top;
    (*target).ptMaxSize.x = info.rcWork.right - info.rcWork.left;
    (*target).ptMaxSize.y = info.rcWork.bottom - info.rcWork.top;
}

unsafe fn mouse_button(
    hwnd: HWND,
    state: *mut WindowState,
    button: MouseButton,
    pressed: bool,
    lparam: LPARAM,
) {
    if state.is_null() {
        return;
    }

    let mask = button_mask(button);
    if pressed {
        (*state).buttons |= mask;
        SetCapture(hwnd);
    } else {
        (*state).buttons &= !mask;
        if (*state).buttons == 0 {
            ReleaseCapture();
        }
    }

    let (x, y) = client_point(lparam);
    (*state).events.push(Event::MouseButton {
        button,
        pressed,
        x,
        y,
        mods: modifiers(),
    });
}

unsafe fn double_click(
    hwnd: HWND,
    state: *mut WindowState,
    button: MouseButton,
    lparam: LPARAM,
) {
    mouse_button(hwnd, state, button, true, lparam);
    if state.is_null() {
        return;
    }
    let (x, y) = client_point(lparam);
    (*state).events.push(Event::MouseDoubleClick { button, x, y });
}

unsafe fn text_input(state: *mut WindowState, unit: u16) {
    if state.is_null() {
        return;
    }

    if (0xD800..=0xDBFF).contains(&unit) {
        (*state).high_surrogate = Some(unit);
        return;
    }

    let scalar = if (0xDC00..=0xDFFF).contains(&unit) {
        let high = match (*state).high_surrogate.take() {
            Some(value) => value,
            None => return,
        };
        0x1_0000 + (((high as u32 - 0xD800) << 10) | (unit as u32 - 0xDC00))
    } else {
        (*state).high_surrogate = None;
        unit as u32
    };

    if let Some(ch) = char::from_u32(scalar) {
        (*state).events.push(Event::Text(ch));
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let state = if message == WM_NCCREATE {
        let create = lparam as *const CREATESTRUCTW;
        if create.is_null() {
            std::ptr::null_mut()
        } else {
            let state = (*create).lpCreateParams as *mut WindowState;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, state as LONG_PTR);
            state
        }
    } else {
        GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState
    };

    match message {
        WM_NCCALCSIZE => 0,

        WM_NCHITTEST => resize_hit(hwnd, lparam),

        WM_GETMINMAXINFO => {
            constrain_maximized(hwnd, lparam);
            0
        }

        WM_SIZE => {
            if !state.is_null() {
                let bits = lparam as u32;
                let width = (bits & 0xFFFF) as u32;
                let height = ((bits >> 16) & 0xFFFF) as u32;

                (*state).width = width;
                (*state).height = height;
                if wparam == SIZE_RESTORED {
                    (*state).normal_width = width;
                    (*state).normal_height = height;
                }
                if wparam != SIZE_MINIMIZED {
                    (*state).maximized = wparam == SIZE_MAXIMIZED;
                }

                (*state).events.push(Event::Resized { width, height });
            }
            0
        }

        WM_DPICHANGED => {
            if !state.is_null() {
                let dpi = (wparam as u32 & 0xFFFF).max(USER_DEFAULT_SCREEN_DPI);
                (*state).dpi = dpi;
            }

            let suggested = lparam as *const RECT;
            if !suggested.is_null() {
                let rect = *suggested;
                SetWindowPos(
                    hwnd,
                    std::ptr::null_mut(),
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    SWP_NOZORDER | SWP_NOACTIVATE,
                );
            }
            0
        }

        WM_SETFOCUS | WM_KILLFOCUS => {
            if !state.is_null() {
                let focused = message == WM_SETFOCUS;
                if !focused {
                    (*state).buttons = 0;
                    ReleaseCapture();
                }
                (*state).events.push(Event::Focus(focused));
            }
            0
        }

        WM_MOUSEMOVE => {
            if !state.is_null() {
                if !(*state).tracking_mouse {
                    let mut track = TRACKMOUSEEVENT {
                        cbSize: std::mem::size_of::<TRACKMOUSEEVENT>() as u32,
                        dwFlags: TME_LEAVE,
                        hwndTrack: hwnd,
                        dwHoverTime: HOVER_DEFAULT,
                    };
                    if TrackMouseEvent(&mut track) != 0 {
                        (*state).tracking_mouse = true;
                    }
                }
                let (x, y) = client_point(lparam);
                (*state).events.push(Event::MouseMove { x, y, mods: modifiers() });
            }
            0
        }

        WM_MOUSELEAVE => {
            if !state.is_null() {
                (*state).tracking_mouse = false;
                (*state).events.push(Event::MouseLeave);
            }
            0
        }

        WM_LBUTTONDOWN => {
            mouse_button(hwnd, state, MouseButton::Left, true, lparam);
            0
        }
        WM_LBUTTONUP => {
            mouse_button(hwnd, state, MouseButton::Left, false, lparam);
            0
        }
        WM_LBUTTONDBLCLK => {
            double_click(hwnd, state, MouseButton::Left, lparam);
            0
        }
        WM_RBUTTONDOWN => {
            mouse_button(hwnd, state, MouseButton::Right, true, lparam);
            0
        }
        WM_RBUTTONUP => {
            mouse_button(hwnd, state, MouseButton::Right, false, lparam);
            0
        }
        WM_RBUTTONDBLCLK => {
            double_click(hwnd, state, MouseButton::Right, lparam);
            0
        }
        WM_MBUTTONDOWN => {
            mouse_button(hwnd, state, MouseButton::Middle, true, lparam);
            0
        }
        WM_MBUTTONUP => {
            mouse_button(hwnd, state, MouseButton::Middle, false, lparam);
            0
        }
        WM_MBUTTONDBLCLK => {
            double_click(hwnd, state, MouseButton::Middle, lparam);
            0
        }

        WM_XBUTTONDOWN | WM_XBUTTONUP | WM_XBUTTONDBLCLK => {
            let button = if high_word(wparam) == XBUTTON2 {
                MouseButton::X2
            } else {
                MouseButton::X1
            };
            if message == WM_XBUTTONDBLCLK {
                double_click(hwnd, state, button, lparam);
            } else {
                mouse_button(hwnd, state, button, message == WM_XBUTTONDOWN, lparam);
            }
            TRUE as LRESULT
        }

        WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
            if !state.is_null() {
                let (x, y) = screen_point(hwnd, lparam);
                let delta = high_word(wparam) as i16 as f32 / WHEEL_DELTA;
                let (delta_y, delta_x) =
                    if message == WM_MOUSEWHEEL { (delta, 0.0) } else { (0.0, delta) };
                (*state).events.push(Event::MouseWheel {
                    delta_y,
                    delta_x,
                    x,
                    y,
                    mods: modifiers(),
                });
            }
            0
        }

        WM_KEYDOWN | WM_SYSKEYDOWN => {
            if !state.is_null() {
                (*state).events.push(Event::Key {
                    key: key_from_virtual(wparam as u32),
                    pressed: true,
                    repeat: lparam as usize & (1usize << 30) != 0,
                    mods: modifiers(),
                });
            }
            if message == WM_SYSKEYDOWN {
                DefWindowProcW(hwnd, message, wparam, lparam)
            } else {
                0
            }
        }

        WM_KEYUP | WM_SYSKEYUP => {
            if !state.is_null() {
                (*state).events.push(Event::Key {
                    key: key_from_virtual(wparam as u32),
                    pressed: false,
                    repeat: false,
                    mods: modifiers(),
                });
            }
            if message == WM_SYSKEYUP {
                DefWindowProcW(hwnd, message, wparam, lparam)
            } else {
                0
            }
        }

        WM_CHAR => {
            text_input(state, wparam as u16);
            0
        }

        WM_SETCURSOR => {
            if !state.is_null() && (lparam as usize & 0xFFFF) as LRESULT == HTCLIENT {
                SetCursor(cursor_handle((*state).cursor));
                return TRUE as LRESULT;
            }
            DefWindowProcW(hwnd, message, wparam, lparam)
        }

        WM_CLOSE => {
            DestroyWindow(hwnd);
            0
        }

        WM_DESTROY => {
            if !state.is_null() {
                (*state).closed = true;
            }
            PostQuitMessage(0);
            0
        }

        WM_ERASEBKGND => 1,

        WM_NCDESTROY => {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            DefWindowProcW(hwnd, message, wparam, lparam)
        }

        _ => DefWindowProcW(hwnd, message, wparam, lparam),
    }
}