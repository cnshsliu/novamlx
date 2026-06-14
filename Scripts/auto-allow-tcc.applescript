#!/usr/bin/osascript
-- Auto-dismiss macOS TCC / privacy prompts that ask permission for NovaMLX
-- (or this script host) to access external volumes like /Volumes/WD.
--
-- Polls every 2s for visible windows that look like TCC prompts and
-- mention our app. When found, clicks the affirmative button (Allow/OK).
--
-- Run via: osascript scripts/auto-allow-tcc.applescript
-- Install via: scripts/install-tcc-watcher.sh install
--
-- Accessibility permission required for the osascript host. Grant once
-- in System Settings -> Privacy & Security -> Accessibility.

use framework "Foundation"
use scripting additions

-- Verbose mode: log every iteration + every window we touch. Useful
-- when diagnosing permission issues; set to false for normal use to
-- reduce log noise (only clicks + errors will be logged).
property VERBOSE : false

on run
    my logMsg("auto-allow-tcc: started; pid=" & (do shell script "echo $$"))
    set pollCount to 0

    set tccHosts to {"UserNotificationCenter", "sharingd", "osascript", "coreauthd", "SecurityAgent", "System Events"}

    repeat
        set pollCount to pollCount + 1
        if VERBOSE or (pollCount mod 30) is 0 then
            my logMsg("auto-allow-tcc: poll=" & (pollCount as text) & " begin")
        end if

        repeat with procName in tccHosts
            try
                tell application "System Events"
                    -- Use EVERY process (not `first`) because multiple processes
                    -- can share a name — e.g. when the watcher is itself an
                    -- osascript and another osascript is showing a `display
                    -- dialog`. `first` returns one of them arbitrarily and may
                    -- pick the watcher itself (no windows), causing us to miss
                    -- the visible dialog owned by the sibling process.
                    try
                        set procList to (every application process whose name is procName)
                    on error
                        set procList to {}
                    end try
                    if VERBOSE then
                        set pc to 0
                        try
                            set pc to count of procList
                        end try
                        my logMsg("auto-allow-tcc: '" & procName & "' procs=" & (pc as text))
                    end if
                    repeat with theProc in procList
                        set wins to {}
                        try
                            set wins to windows of theProc
                        end try
                        if VERBOSE and (count of wins) > 0 then
                            my logMsg("auto-allow-tcc: '" & procName & "' has " & ((count of wins) as text) & " window(s)")
                        end if
                        repeat with w in wins
                            try
                                set winName to ""
                                try
                                    set winName to (name of w) as text
                                end try
                                set winDesc to ""
                                try
                                    set winDesc to (description of w) as text
                                end try
                                set bodyText to ""
                                try
                                    set staticList to (static texts of w)
                                    repeat with aStatic in staticList
                                        try
                                            set bodyText to bodyText & ((value of aStatic) as text) & " "
                                        end try
                                    end repeat
                                end try

                                if VERBOSE then
                                    my logMsg("auto-allow-tcc: win '" & winName & "' desc='" & winDesc & "' body='" & bodyText & "'")
                                end if

                                set combined to winName & " " & winDesc & " " & bodyText
                                set combinedLower to my lowerCase(combined)

                                set looksLikeTCC to false
                                repeat with m in {"wants to access", "would like to access", "files in", "files on", "removable volume", "external volume", "documents folder", "downloads folder", "desktop folder"}
                                    if combinedLower contains m then
                                        set looksLikeTCC to true
                                        exit repeat
                                    end if
                                end repeat

                                set mentionsApp to false
                                repeat with m in {"novamlx", "nova", "osascript", "terminal"}
                                    if combinedLower contains m then
                                        set mentionsApp to true
                                        exit repeat
                                    end if
                                end repeat

                                if looksLikeTCC and mentionsApp then
                                    set clicked to false
                                    try
                                        set btnList to (buttons of w)
                                        repeat with b in btnList
                                            set bName to ""
                                            try
                                                set bName to (name of b) as text
                                            end try
                                            set bLower to my lowerCase(bName)
                                            if bLower is "allow" or bLower is "ok" or bLower is "always allow" or bLower is "continue" or bLower is "allow anyway" then
                                                click b
                                                set clicked to true
                                                my logMsg("auto-allow-tcc: CLICKED '" & bName & "' in '" & winName & "' (process=" & procName & ")")
                                                exit repeat
                                            end if
                                        end repeat
                                    end try
                                    if not clicked then
                                        my logMsg("auto-allow-tcc: matched dialog but no Allow button: '" & winName & "' body=" & bodyText)
                                    end if
                                end if
                            on error errMsg number errNum
                                my logMsg("auto-allow-tcc: inner err " & (errNum as text) & ": " & errMsg)
                            end try
                        end repeat
                    end repeat
                end tell
            on error errMsg number errNum
                my logMsg("auto-allow-tcc: proc '" & procName & "' error " & (errNum as text) & ": " & errMsg)
            end try
        end repeat

        if VERBOSE then
            my logMsg("auto-allow-tcc: poll=" & (pollCount as text) & " end")
        end if

        delay 2
    end repeat
end run

on lowerCase(s)
    return (current application's NSString's stringWithString:s)'s lowercaseString() as text
end lowerCase

on logMsg(msg)
    try
        do shell script "mkdir -p \"$HOME/.nova/logs\"; echo \"$(date '+%Y-%m-%d %H:%M:%S') " & msg & "\" >> \"$HOME/.nova/logs/tcc-watcher.log\""
    end try
end logMsg
