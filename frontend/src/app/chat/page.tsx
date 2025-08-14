'use client';

import { useState, FormEvent, useRef, useEffect, useCallback } from 'react';
import { useUser } from '../../context/UserContext';
import { usePathname, useRouter } from 'next/navigation';

interface Message {
    sender: string;
    model: string;
    text: string;
}

// Define the shape of the typing state for our agents
interface TypingState {
    [key: string]: boolean;
}

interface ServerMessage {
  sender?: string;
  text?: string;
  typing?: boolean;
  // include conversation id messages
  type?: 'ping' | 'pong' | 'conversation_id' | string;
  id?: string; // used when type === 'conversation_id'
  summary?: string;       // when type === 'context_summary'
}

// Past-convo list item returned by the API
interface ConversationListItem {
  id: string;
  title: string;
  updated_at?: string;
}

// put this just after your interfaces
type HBWebSocket = WebSocket & { _heartbeatInterval?: number };

// Base REST API (same host as WS but https/http, not ws)
const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, '')) ||
  'https://api.aicolosseum.app';

// Helper: fetch with Authorization header and 1x retry on 401 using /api/refresh
async function apiFetch(pathOrUrl: string | URL, init: RequestInit = {}) {
  // Always build absolute URL against API_BASE
  const url =
    typeof pathOrUrl === 'string'
      ? (pathOrUrl.startsWith('http')
          ? pathOrUrl
          : `${API_BASE}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`)
      : pathOrUrl.toString();

  const headers = new Headers(init.headers || {});
  const access = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  if (access) headers.set('Authorization', `Bearer ${access}`);

  // always send cookies so /api/refresh can read the refresh cookie
  let res = await fetch(url, { ...init, headers, credentials: 'include' });
  if (res.status !== 401) return res;

  // try to refresh access token
  const rr = await fetch(`${API_BASE}/api/refresh`, {
    method: 'POST',
    credentials: 'include',
  });
  if (!rr.ok) return res;

  const { token } = await rr.json();
  if (token) {
    if (typeof window !== 'undefined') localStorage.setItem('access_token', token);
    headers.set('Authorization', `Bearer ${token}`);
    res = await fetch(url, { ...init, headers, credentials: 'include' });
  }
  return res;
}


export default function ChatPage() {
    const { userName, userToken } = useUser();
    const router = useRouter();
    const pathname = usePathname();

    const [message, setMessage] = useState<string>('');
    const [chatHistory, setChatHistory] = useState<Message[]>([]);
    const [isWsOpen, setIsWsOpen] = useState<boolean>(false);
    // Shows a one-time banner when a past-session summary exists
    const [loadedSummary, setLoadedSummary] = useState<string | null>(null);

    // Toggle to reveal/hide the text of the summary
    const [showSummary, setShowSummary] = useState(false);

    const [wsReconnectNonce, setWsReconnectNonce] = useState(0);
    const [conversationId, setConversationId] = useState<string | null>(null);

    // Sidebar + conversation list
    const [conversations, setConversations] = useState<ConversationListItem[]>([]);
    const [isLoadingConvs, setIsLoadingConvs] = useState(false);

    // New state to track which agents are typing
    const [isTyping, setIsTyping] = useState<TypingState>({
        ChatGPT: false,
        Claude: false,
        Gemini: false,
        Mistral: false,
    });

    // A simple ref to distinguish the first render
    const isInitialRender = useRef(true);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const chatEndRef = useRef<HTMLDivElement>(null);
    const ws = useRef<WebSocket | null>(null);

    // ws reconnect guards/backoff
    const authFailedRef = useRef(false);
    const reconnectBackoffRef = useRef(1000); // start at 1s, exponential up to 15s

    // --- guarded token refresh helpers ---
    const refreshInFlight = useRef<Promise<void> | null>(null);
    const lastRefreshAt = useRef<number>(0);

    async function refreshTokenIfNeeded(force = false) {
    const now = Date.now();
    // throttle non-forced refresh to at most once/minute
    if (!force && now - lastRefreshAt.current < 60_000) return;
    if (refreshInFlight.current) return refreshInFlight.current;

    refreshInFlight.current = (async () => {
        try {
        const res = await fetch(`${API_BASE}/api/refresh`, {
            method: 'POST',
            credentials: 'include',
        });
        if (!res.ok) throw new Error('refresh failed');
        const { token } = await res.json();
        if (token) {
            localStorage.setItem('access_token', token);
            lastRefreshAt.current = Date.now();
        } else {
            throw new Error('no token in refresh');
        }
        } finally {
        // always clear the in-flight marker
        refreshInFlight.current = null;
        }
    })();

    return refreshInFlight.current;
    }
    // --- end guarded token refresh helpers ---


    // Unconditional scroll to the last message
    useEffect(() => {
        if (isInitialRender.current) {
            isInitialRender.current = false;
            return;
        }
        if (chatEndRef.current) {
            const behavior = 'smooth';
            chatEndRef.current.scrollIntoView({ behavior });
        }
    }, [chatHistory]);
    
    // load saved conversation id on mount
    useEffect(() => {
        try {
            const saved = localStorage.getItem('conversationId');
            if (saved) setConversationId(saved);
        } catch {}
        }, []);

        // save id whenever it changes
        useEffect(() => {
        try {
            if (conversationId) localStorage.setItem('conversationId', conversationId);
        } catch {}
    }, [conversationId]);

    // Fetch the list of conversations
    const loadConversations = useCallback(async () => {
    if (!userToken) return;
        try {
            setIsLoadingConvs(true);
            const res = await apiFetch(`${API_BASE}/api/conversations?limit=100`, {            
            cache: 'no-store',
            });
            if (!res.ok) throw new Error(`List convos failed: ${res.status}`);
            const data = await res.json();
            // Support either {items:[...]} or [...] responses
            setConversations(Array.isArray(data) ? data : (data.items ?? []));
        } catch (e) {
            console.warn('loadConversations error:', e);
        } finally {
            setIsLoadingConvs(false);
        }
    }, [userToken]);

    // keep the sidebar list fresh
    useEffect(() => {
        if (userToken) loadConversations();
    }, [userToken, conversationId, loadConversations]);

    // Handle redirection to sign-in page when userToken is not present
    useEffect(() => {
    if (!userToken) {
        // wipe conversation continuity on sign-out
        try { localStorage.removeItem('conversationId'); } catch {}
        setConversationId(null);
        setChatHistory([]);
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });

        // close any open socket
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
        ws.current.close(1000, 'logout');
        }
        ws.current = null;

        // route to sign-in if not already there
        if (pathname !== '/sign-in') {
        router.push('/sign-in');
        }
        return; // don’t try to connect without a token
    }
    }, [userToken, pathname, router]);


    // Use useCallback to memoize the function, preventing unnecessary re-renders
    const addMessageToChat = useCallback((msg: { sender: string; text: string }) => {
        const cleanSender = msg.sender.trim();
        
        setChatHistory(prev => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage && lastMessage.sender === cleanSender) {
                return prev.map((item, index) => 
                    index === prev.length - 1 ? { ...item, text: `${item.text}\n${msg.text}` } : item
                );
            }
            return [...prev, { sender: cleanSender, model: cleanSender, text: msg.text }];
        });
    }, []);

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        // Updated condition
        if (!message.trim() || !ws.current || ws.current.readyState !== WebSocket.OPEN) return;
        
        const userMessage = { sender: userName || 'You', model: 'Human User', text: message };
        setChatHistory(prev => [...prev, userMessage]);
        
        // Prepare the payload for the backend with Mistral flag
        const payload = {
            message,
            user_name: userName,
        };
        ws.current.send(JSON.stringify(payload));
        
        setMessage('');
    };








    // Open an existing conversation from the sidebar
    const handleOpenConversation = async (id: string) => {
        try { localStorage.setItem('conversationId', id); } catch {}
        setConversationId(id);
        setLoadedSummary(null);
        setShowSummary(false);
        setChatHistory([]);
        setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });

        // Reconnect WS so the server seeds the prior context
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            ws.current.close(1000, 'switch-conversation');
        }
        ws.current = null;
        setTimeout(() => setWsReconnectNonce(n => n + 1), 50);
    };

    // Create a brand-new conversation
    const handleNewConversation = () => {
        try { localStorage.removeItem('conversationId'); } catch {}
        setConversationId(null);
        setLoadedSummary(null);
        setShowSummary(false);
        setChatHistory([]);
        setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            ws.current.close(1000, 'new-conversation');
        }
        ws.current = null;
        setTimeout(() => setWsReconnectNonce(n => n + 1), 50);
    };

    // Rename the selected conversation
    const handleRenameConversation = async (id: string) => {
        if (!userToken) return;

        const current = conversations.find((c) => c.id === id);
        const proposed = window.prompt('Rename conversation to:', current?.title ?? '');
        if (!proposed || !proposed.trim()) return;

        const res = await apiFetch(`${API_BASE}/api/conversations/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: proposed.trim() }),
        });
        if (!res.ok) {
            alert('Rename failed.');
            return;
        }
        await loadConversations();
    };

    // Delete a conversation
    const handleDeleteConversation = async (id: string) => {
        if (!userToken) return;
        if (!window.confirm('Delete this conversation? This cannot be undone.')) return;

        const res = await apiFetch(`${API_BASE}/api/conversations/${id}`, {
            method: 'DELETE',            
        });
        if (!res.ok) {
            alert('Delete failed.');
            return;
        }

        // If we deleted the one we’re viewing, start fresh
        if (conversationId === id) {
            handleNewConversation();
        } else {
            await loadConversations();
        }
    };


    // WebSocket connection logic
    useEffect(() => {
    // don’t try to connect without a token
    if (!userToken) {
        if (ws.current) {
            try { ws.current.close(); } catch {}
        }
        ws.current = null;
        setIsWsOpen(false);
        return;
    }
    // try to top up access token if it's close to expiring (non-blocking)
    refreshTokenIfNeeded();


    // Build the URL safely
    const base = process.env.NEXT_PUBLIC_WS_URL ?? 'http://localhost:8000';
    let u: URL;
    try {
        u = new URL(base); // e.g. https://api.aicolosseum.app
    } catch {
        u = new URL('http://localhost:8000');
    }
    u.protocol = (u.protocol === 'https:') ? 'wss:' : 'ws:';
    u.pathname = '/ws/colosseum-chat';
    u.search = `?token=${encodeURIComponent(localStorage.getItem('access_token') || userToken)}`;

    const socket = new WebSocket(u.toString());
    ws.current = socket;
    setIsWsOpen(false);

    const currentWs = socket;

    currentWs.onopen = () => {
        setIsWsOpen(true);
        authFailedRef.current = false;
        reconnectBackoffRef.current = 1000;

        const initialPayload: Record<string, unknown> = {
        message: 'Hello!',
        user_name: userName,
        };
        if (conversationId) {
            initialPayload.conversation_id = conversationId; // reuse exact convo
        } else {
            initialPayload.resume_last = true;               // ask server to resume latest
        }

        currentWs.send(JSON.stringify(initialPayload));
    };

    currentWs.onmessage = (event: MessageEvent) => {
        let msg: ServerMessage;
        try { msg = JSON.parse(event.data); }
        catch { return; }

        // meta/system messages
        if (msg.type === 'conversation_id' && typeof msg.id === 'string') {
        setConversationId(msg.id);
        loadConversations(); // refresh sidebar
        return;
        }
        if (msg.type === 'context_summary' && typeof msg.summary === 'string' && msg.summary.trim()) {
        setLoadedSummary(msg.summary);
        return;
        }
        if (msg.type === 'ping' || msg.type === 'pong') return;

        // typing
        if (typeof msg.sender === 'string' && typeof msg.typing === 'boolean') {
            const key = msg.sender;      // narrowed to string
            const val = msg.typing;      // narrowed to boolean
            setIsTyping(prev => {
                const next: TypingState = { ...prev };
                next[key] = val;
                return next;
            });
            return;
        }

        // normal chat
        if (typeof msg.sender === 'string' && typeof msg.text === 'string') {
        addMessageToChat({ sender: msg.sender, text: msg.text });
        return;
        }
    };

    currentWs.onclose = async (ev) => {
        console.log('WebSocket closed. code=', ev.code, 'reason=', ev.reason, 'wasClean=', ev.wasClean);
        setIsWsOpen(false);
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });


        // mark closed
        ws.current = null;

        // If it's an auth-related closure, try a SINGLE forced refresh
        const AUTHISH = ev.code === 1008 || ev.code === 4001 || ev.code === 4002 || ev.code === 4003;
        if (AUTHISH && !authFailedRef.current) {
            try {
            await refreshTokenIfNeeded(true); // force refresh
            authFailedRef.current = false;    // success → keep reconnecting
            reconnectBackoffRef.current = 1000;
            } catch {
            // refresh truly failed → STOP reconnecting to avoid loops
            authFailedRef.current = true;
            }
        }

        if (authFailedRef.current) {
            console.warn('Auth refresh failed; not auto-reconnecting.');
            return;
        }

        // Exponential backoff reconnect
        const delay = reconnectBackoffRef.current;
        reconnectBackoffRef.current = Math.min(reconnectBackoffRef.current * 2, 15000);
        // Don't reconnect while tab is hidden to avoid loops on background tabs
        if (document.hidden) {
        const onVisible = () => {
            document.removeEventListener('visibilitychange', onVisible);
            setWsReconnectNonce(n => n + 1);
        };
        document.addEventListener('visibilitychange', onVisible);
        return;
        }

        setTimeout(() => setWsReconnectNonce(n => n + 1), delay);
    };


    currentWs.onerror = (event: Event) => {
        console.error('WebSocket error:', event);
        setIsWsOpen(false);
    };

    return () => {
        if (currentWs && (currentWs.readyState === WebSocket.OPEN || currentWs.readyState === WebSocket.CONNECTING)) {
        try { currentWs.close(); } catch {}
        }
    };
    }, [userToken, userName, conversationId, addMessageToChat, loadConversations, wsReconnectNonce]);

        

    // keep access token fresh while user is signed in
    const refreshTimerId = useRef<number | null>(null);

    useEffect(() => {
    if (!userToken) {
        // clear any existing timer if user logs out
        if (refreshTimerId.current) {
        window.clearInterval(refreshTimerId.current);
        refreshTimerId.current = null;
        }
        return;
    }

    // do an initial (throttled) refresh to extend session
    refreshTokenIfNeeded().catch(() => {});

    // set interval
    if (refreshTimerId.current) window.clearInterval(refreshTimerId.current);
    refreshTimerId.current = window.setInterval(() => {
        refreshTokenIfNeeded().catch(() => {});
    }, 5 * 60 * 1000);

    // cleanup
    return () => {
        if (refreshTimerId.current) {
        window.clearInterval(refreshTimerId.current);
        refreshTimerId.current = null;
        }
    };
    }, [userToken]);


    
    const handleResetConversation = () => {
        try { localStorage.removeItem('conversationId'); } catch {}
        setConversationId(null);
        setLoadedSummary(null);          // <- add this line
        setShowSummary(false);           // <- and this line
        setChatHistory([]);
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });

        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            ws.current.close(1000, 'user-reset');
        }
        ws.current = null;

        setTimeout(() => setWsReconnectNonce(n => n + 1), 50);
    };





    return (
    <div className="flex h-screen w-full bg-gray-950 text-white font-sans antialiased">
        {/* LEFT SIDEBAR */}
        <aside className="hidden md:flex flex-col w-72 border-r border-gray-800 bg-gray-900">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <div className="font-semibold">Conversations</div>
            <button
            onClick={handleNewConversation}
            className="text-xs px-2 py-1 rounded bg-blue-600 hover:bg-blue-700"
            >
            New
            </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
            {isLoadingConvs && (
            <div className="p-4 text-sm text-gray-400">Loading…</div>
            )}
            {!isLoadingConvs && conversations.length === 0 && (
            <div className="p-4 text-sm text-gray-400">No conversations yet</div>
            )}
            <ul className="divide-y divide-gray-800">
            {conversations.map((c) => (
                <li
                key={c.id}
                className={`group px-3 py-2 cursor-pointer ${
                    conversationId === c.id
                    ? 'bg-gray-800'
                    : 'hover:bg-gray-800/60'
                }`}
                >
                <div className="flex items-center justify-between gap-2">
                    <button
                    onClick={() => handleOpenConversation(c.id)}
                    className="flex-1 text-left truncate"
                    title={c.title}
                    >
                    {c.title || 'Untitled'}
                    </button>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                    <button
                        onClick={() => handleRenameConversation(c.id)}
                        className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
                        title="Rename"
                    >
                        Rename
                    </button>
                    <button
                        onClick={() => handleDeleteConversation(c.id)}
                        className="text-xs px-2 py-1 rounded bg-red-700 hover:bg-red-600"
                        title="Delete"
                    >
                        Delete
                    </button>
                    </div>
                </div>
                {c.updated_at && (
                    <div className="mt-1 text-[11px] text-gray-400">
                    {new Date(c.updated_at).toLocaleString()}
                    </div>
                )}
                </li>
            ))}
            </ul>
        </div>
        </aside>

        {/* MAIN COLUMN (YOUR EXISTING CHAT UI, UNCHANGED) */}
        <div className="flex-1 flex flex-col">
        <main
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto pt-[72px] pb-[100px] bg-gray-900 custom-scrollbar"
        >
            <div className="max-w-4xl mx-auto flex flex-col space-y-4 md:space-y-6">
            {/* Conditional rendering for the initial message */}
            {!isWsOpen && userToken ? (
                <p className="text-center text-gray-500 py-12 text-lg">
                Connecting to chat...
                </p>
            ) : (
                chatHistory.length === 0 && (
                <p className="text-center text-gray-500 py-12 text-lg">
                    {userName
                    ? `Start a conversation with the AI team, ${userName}...`
                    : `Please sign in to start a conversation...`}
                </p>
                )
            )}

            {chatHistory.length > 0 &&
                chatHistory.map((msg, index) => (
                <div
                    key={index}
                    className={`flex ${
                    msg.sender === (userName || 'You')
                        ? 'justify-end'
                        : 'justify-start'
                    }`}
                >
                    <div
                    className={`relative p-3 md:p-4 max-w-[80%] text-white rounded-2xl md:rounded-3xl shadow-lg transition-all duration-200 ease-in-out transform hover:scale-[1.01] ${
                        msg.sender === (userName || 'You')
                        ? 'bg-blue-600 text-white rounded-tr-none'
                        : msg.sender === 'Claude'
                        ? 'bg-gray-700 text-white rounded-bl-none'
                        : msg.sender === 'ChatGPT'
                        ? 'bg-gray-800 text-white rounded-bl-none'
                        : msg.sender === 'Gemini'
                        ? 'bg-gray-600 text-white rounded-bl-none'
                        : msg.sender === 'Mistral'
                        ? 'bg-gray-500 text-white rounded-bl-none'
                        : 'bg-gray-700 text-gray-200 rounded-bl-none'
                    }`}
                    >
                    {msg.sender !== (userName || 'You') && (
                        <div className="text-xs text-gray-300 mb-1 font-semibold">
                        {msg.sender}
                        </div>
                    )}
                    <pre className="whitespace-pre-wrap font-sans text-sm md:text-base leading-relaxed">
                        {msg.text}
                    </pre>
                    </div>
                </div>
                ))}

            <div className="flex flex-col space-y-2">
                {isTyping.ChatGPT && (
                <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-gray-800 text-white shadow-md">
                    <div className="flex space-x-1">
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                    </div>
                    <span className="text-sm text-gray-300 font-medium">
                    ChatGPT is typing...
                    </span>
                </div>
                )}
                {isTyping.Claude && (
                <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-gray-700 text-white shadow-md">
                    <div className="flex space-x-1">
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                    </div>
                    <span className="text-sm text-gray-300 font-medium">
                    Claude is typing...
                    </span>
                </div>
                )}
                {isTyping.Gemini && (
                <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-purple-800 text-white shadow-md">
                    <div className="flex space-x-1">
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                    </div>
                    <span className="text-sm text-gray-300 font-medium">
                    Gemini is typing...
                    </span>
                </div>
                )}
                {isTyping.Mistral && (
                <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-green-800 text-white shadow-md">
                    <div className="flex space-x-1">
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                    </div>
                    <span className="text-sm text-gray-300 font-medium">
                    Mistral is typing...
                    </span>
                </div>
                )}
            </div>

            <div ref={chatEndRef} />
            </div>
        </main>

        {/* INPUT FORM — fixed on mobile, static on desktop so it doesn't cover the sidebar */}
        <form
            onSubmit={handleSubmit}
            className="fixed md:static bottom-0 left-0 right-0 z-10 bg-gray-900 border-t border-gray-800 p-4 md:p-6 shadow-lg"
        >
            {loadedSummary && (
            <div className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                <div className="flex items-start justify-between gap-3">
                <div>
                    <strong>Context loaded</strong> — this chat includes knowledge from earlier sessions.
                </div>
                <button
                    type="button"
                    className="shrink-0 underline"
                    onClick={() => setShowSummary((s) => !s)}
                >
                    {showSummary ? 'Hide' : 'Show'} summary
                </button>

                </div>
                {showSummary && (
                <div className="mt-2 whitespace-pre-wrap">{loadedSummary}</div>
                )}
            </div>
            )}

            <div className="flex gap-4 max-w-4xl mx-auto">
            <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type your message..."
                className="flex-grow p-3 rounded-xl border border-gray-700 bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 text-sm md:text-base"
                disabled={!isWsOpen || !userName}
            />
            <button
                type="submit"
                className="px-4 py-2 md:px-6 md:py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 transition-colors duration-200 text-sm"
                disabled={!isWsOpen || !userName}
            >
                Send
            </button>
            <button
                type="button"
                onClick={handleResetConversation}
                className="px-4 py-2 md:px-6 md:py-3 bg-gray-700 text-white font-semibold rounded-xl hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-gray-900 transition-colors duration-200 text-sm"
                disabled={!isWsOpen}
            >
                Reset conversation
            </button>
            </div>
        </form>
        </div>
    </div>
    );

}