'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useUser } from '../../context/UserContext';

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
  sender?: string | null;
  text?: string;
  typing?: boolean;
  type?: 'ping' | 'pong' | 'conversation_id' | 'context_summary' | 'conversation_meta' | string;
  id?: string;
  summary?: string;

  // add these for conversation_meta payloads
  title?: string;
  updated_at?: string;
}

// Past-convo list item returned by the API
interface ConversationListItem {
  id: string;
  title: string;
  updated_at?: string;
  created_at?: string | null;   // ← NEW (for fallback display)
}

// Typed shape used when sorting/choosing newest in preselects
type SidebarConvo = ConversationListItem;

interface StoredMessage {
  sender: string;
  content: string;
  role?: string;
  created_at?: string;
}

/** Helper types to avoid `any` while staying flexible */
type ContentFragment = string | { text?: string } | { type?: string; text?: string } | unknown[];

interface BackendMessageLike {
  sender?: string;
  agent?: string;
  role?: string;
  model?: string;
  content?: unknown;
  text?: string;
  message?: string;
  body?: unknown;
}

type ConversationListResponse =
  | ConversationListItem[]
  | { items: ConversationListItem[] }
  | { conversations: ConversationListItem[] }
  | { data: ConversationListItem[] }
  | { data: { items: ConversationListItem[] } };

type UploadedAttachment = {
  id: string;
  name: string;
  mime: string;
  size: number;
  signed_url?: string | null;
};


// Base REST API (mirror WS host but force http/https for REST)
function resolveApiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, '');
  if (fromEnv) return fromEnv;

  const wsBase = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
  try {
    const u = new URL(wsBase);
    // Flip any ws/wss to http/https for REST
    if (u.protocol === 'ws:') u.protocol = 'http:';
    else if (u.protocol === 'wss:') u.protocol = 'https:';
    u.pathname = '';
    u.search = '';
    u.hash = '';
    return u.toString().replace(/\/+$/, '');
  } catch {
    return 'http://localhost:8000';
  }
}
const API_BASE = resolveApiBase();

// Always include an Authorization header using either localStorage or userToken fallback.
// We'll pass this from handlers that need guaranteed auth even if localStorage isn't ready yet.
function buildAuthHeaders(userToken?: string | null): Headers {
  const h = new Headers({ 'Content-Type': 'application/json' });
  const local = (typeof window !== 'undefined') ? localStorage.getItem('access_token') : null;
  // normalize null/undefined to an empty string for the header check
  const tok = (local ?? userToken) ?? '';
  if (tok) h.set('Authorization', `Bearer ${tok}`);
  return h;
}




// Helper: fetch with Authorization header and 1x retry on 401 using /api/refresh
async function apiFetch(pathOrUrl: string | URL, init: RequestInit = {}) {
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

  // Try one refresh
  const rr = await fetch(`${API_BASE}/api/refresh`, {
    method: 'POST',
    credentials: 'include',
  });

  if (!rr.ok) {
    // Refresh failed — clear any stale token and return original 401
    try { localStorage.removeItem('access_token'); } catch {}
    return res;
  }

  const { token } = await rr.json().catch(() => ({ token: null as string | null }));
  if (!token) {
    try { localStorage.removeItem('access_token'); } catch {}
    return res; // still return original response to keep call-sites simple
  }

  // Retry once with the fresh token
  try { localStorage.setItem('access_token', token); } catch {}
  headers.set('Authorization', `Bearer ${token}`);
  res = await fetch(url, { ...init, headers, credentials: 'include' });
  return res;
}

// Always return a usable access token or null (and set it in localStorage if we got a fresh one)
async function getFreshAccessToken(): Promise<string | null> {
  const existing =
    typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  if (existing) return existing;

  try {
    const r = await fetch(`${API_BASE}/api/refresh`, {
      method: 'POST',
      credentials: 'include', // send refresh cookie
    });
    if (!r.ok) return null;
    const { token } = await r.json().catch(() => ({ token: null as string | null }));
    if (!token) return null;
    try { localStorage.setItem('access_token', token); } catch {}
    return token;
  } catch {
    return null;
  }
}

type AgentName = 'ChatGPT' | 'Claude' | 'Gemini' | 'Mistral';
const ALLOWED_AGENTS: AgentName[] = ['ChatGPT', 'Claude', 'Gemini', 'Mistral'];

export default function ChatPage() {

    const typingTTLRef = useRef<Partial<Record<AgentName, number>>>({});
    const typingShowDelayRef = useRef<Partial<Record<AgentName, number>>>({});
    const typingTimersRef = useRef<Partial<Record<AgentName, number>>>({});    
    
    const clearTypingTTL = (agent: AgentName) => {
        const id = typingTTLRef.current[agent];
        if (typeof id === 'number') {
            window.clearTimeout(id);
            delete typingTTLRef.current[agent];
        }
    };
    const clearTypingShowDelay = (agent: AgentName) => {
        const id = typingShowDelayRef.current[agent];
        if (typeof id === 'number') {
            window.clearTimeout(id);
            delete typingShowDelayRef.current[agent];
        }
    };

    /** Only show bubble if the agent is still "typing" after showDelayMs.
     *  When shown, auto-clear after ttlMs unless a message/false arrives. */
    const setTypingWithDelayAndTTL = (agent: AgentName, value: boolean, showDelayMs = 400, ttlMs = 12000) => {
        if (!value) {
            // cancel pending show + ttl, hide immediately
            clearTypingShowDelay(agent);
            clearTypingTTL(agent);
            setIsTyping(prev => ({ ...prev, [agent]: false }));
            return;
        }

        // schedule showing after a short delay
        clearTypingShowDelay(agent);
        typingShowDelayRef.current[agent] = window.setTimeout(() => {
            setIsTyping(prev => ({ ...prev, [agent]: true }));

            // (re)start TTL once it’s actually visible
            clearTypingTTL(agent);
            typingTTLRef.current[agent] = window.setTimeout(() => {
            setIsTyping(prev => ({ ...prev, [agent]: false }));
            delete typingTTLRef.current[agent];
            }, ttlMs);
        }, showDelayMs);
    };    

    const { userName, userToken } = useUser();
    const [message, setMessage] = useState<string>('');
    const [chatHistory, setChatHistory] = useState<Message[]>([]);
    const [isWsOpen, setIsWsOpen] = useState<boolean>(false);
    // Shows a one-time banner when a past-session summary exists
    const [loadedSummary, setLoadedSummary] = useState<string | null>(null);

    // Toggle to reveal/hide the text of the summary
    const [showSummary, setShowSummary] = useState(false);

    const [wsReconnectNonce, setWsReconnectNonce] = useState(0);
    const [conversationId, setConversationId] = useState<string | null>(null);
    
    const [pendingFiles, setPendingFiles] = useState<UploadedAttachment[]>([]);

    const uploadOne = async (file: File, conversationId: string | null): Promise<UploadedAttachment> => {
        const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
        const form = new FormData();
        form.append('file', file);
        if (conversationId) form.append('conversation_id', conversationId);

        const res = await fetch(`${API_BASE}/api/uploads`, {
            method: 'POST',
            body: form,
            headers: token ? { Authorization: `Bearer ${token}` } : undefined,
            credentials: 'include',
        });
        if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
        return res.json();
    };

    const handleFilePick = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || !files.length) return;
        try {
            const limit = Math.min(files.length, 3); // keep in sync with backend
            const uploaded: UploadedAttachment[] = [];
            for (let i = 0; i < limit; i++) {
            uploaded.push(await uploadOne(files[i], conversationId));
            }
            setPendingFiles(prev => [...prev, ...uploaded]);
        } catch (err) {
            console.error(err);
            alert('Upload failed. Try smaller text/code files.');
        } finally {
            // reset the input so the same file can be chosen again
            e.currentTarget.value = '';
        }
    };

    const removePending = (id: string) => setPendingFiles(prev => prev.filter(p => p.id !== id));


    // Sidebar + conversation list
    const [conversations, setConversations] = useState<ConversationListItem[]>([]);
    // Hydrate conversations from localStorage on mount
    useEffect(() => {
        if (typeof window === 'undefined') return; // avoid SSR access
        try {
            const cached = localStorage.getItem('conversations_cache');
            if (cached) {
                const parsed = JSON.parse(cached) as ConversationListItem[];
                if (Array.isArray(parsed)) {
                    setConversations(parsed);
                }
            }
        } catch {
            // ignore parse/storage errors
        }
    }, []);

    // Persist conversations to localStorage whenever they change
    useEffect(() => {
        if (typeof window === 'undefined') return;
        try {
            localStorage.setItem('conversations_cache', JSON.stringify(conversations));
        } catch {
            // ignore storage errors
        }
    }, [conversations]);

    const [isLoadingConvs, setIsLoadingConvs] = useState(false);
    // Sidebar bulk-manage state
    const [manageMode, setManageMode] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const chatContainerRef = useRef<HTMLDivElement | null>(null);
    const composerRef = useRef<HTMLDivElement | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);
    const [composerHeight, setComposerHeight] = useState<number>(120);

    useEffect(() => {
        const el = composerRef.current;
        if (!el || typeof window === 'undefined') return;
        const ro = new ResizeObserver(() => {
            setComposerHeight(el.offsetHeight);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // Nudge scroll when the composer height changes so content never hides behind it
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [composerHeight]);    

    // New state to track which agents are typing
    const [isTyping, setIsTyping] = useState<TypingState>({
        ChatGPT: false,
        Claude: false,
        Gemini: false,
        Mistral: false,
    });

    // A simple ref to distinguish the first render
    const isInitialRender = useRef(true);
    const chatEndRef = useRef<HTMLDivElement>(null);
    const ws = useRef<WebSocket | null>(null);
    const chatLengthRef = useRef(0);
    useEffect(() => { chatLengthRef.current = chatHistory.length; }, [chatHistory.length]);
    const pendingSends = useRef<Array<Record<string, unknown>>>([]);
    // Tracks reconnect backoff and any pending timer
    const reconnectRef = useRef<{ tries: number; timer: number | null }>({
        tries: 0,
        timer: null,
    });


    // ws reconnect guards/backoff
    const authFailedRef = useRef(false);

    // --- guarded token refresh helpers (NO HOOKS INSIDE) ---
    const refreshInFlight = useRef<Promise<string | null> | null>(null);

    const lastRefreshAt = useRef<number>(0);

    const refreshTokenIfNeeded = useCallback(
        async (force = false): Promise<string | null> => {
        const now = Date.now();

        // throttle non-forced refresh to at most once/minute
        if (!force && now - lastRefreshAt.current < 60_000) return null;
        if (refreshInFlight.current) return refreshInFlight.current;

        refreshInFlight.current = (async () => {
            try {
            const res = await fetch(`${API_BASE}/api/refresh`, {
                method: 'POST',
                credentials: 'include',
            });

            if (!res.ok) {
                // 401/expired/etc. -> clear any stale access token and bail
                try { localStorage.removeItem('access_token'); } catch {}
                lastRefreshAt.current = 0;
                return null;
            }

            const { token } = await res.json().catch(() => ({ token: null as string | null }));
            if (token) {
                try { localStorage.setItem('access_token', token); } catch {}
                lastRefreshAt.current = Date.now();
                return token as string;
            }

            // No token in response — treat as unauthenticated
            try { localStorage.removeItem('access_token'); } catch {}
            lastRefreshAt.current = 0;
            return null;
            } finally {
            // always clear the in-flight marker
            refreshInFlight.current = null;
            }
        })();

        return refreshInFlight.current;
    }, []);




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
    

    const hydrateConversation = useCallback(async (id: string) => {
    try {
        // ensure we have a token (apiFetch will still 401->refresh->retry)
        const res = await apiFetch(`/api/conversations/${id}/messages?limit=200`, {
            cache: 'no-store',
            headers: buildAuthHeaders(userToken ?? undefined),
        });
        if (!res.ok) {
            const txt = await res.text().catch(() => '');
            console.warn('hydrate failed', res.status, txt);
            return;
        }

            const data: unknown = await res.json();

            // Tolerate many shapes: [], {items}, {messages}, {data}, {data:{items}}, {results}
            const rawUnknown: unknown =
            Array.isArray(data) ? data
            : Array.isArray((data as { items?: unknown[] })?.items) ? (data as { items: unknown[] }).items
            : Array.isArray((data as { messages?: unknown[] })?.messages) ? (data as { messages: unknown[] }).messages
            : Array.isArray((data as { results?: unknown[] })?.results) ? (data as { results: unknown[] }).results
            : Array.isArray((data as { data?: unknown[] })?.data) ? (data as { data: unknown[] }).data
            : Array.isArray((data as { data?: { items?: unknown[] } })?.data?.items)
                ? ((data as { data: { items: unknown[] } }).data.items)
                : [];

            const items: BackendMessageLike[] = Array.isArray(rawUnknown)
            ? rawUnknown.filter((m): m is BackendMessageLike => typeof m === 'object' && m !== null)
            : [];

            const toText = (c: unknown): string => {
            if (typeof c === 'string') return c;
            if (Array.isArray(c)) {
                return c
                .map((part) => {
                    if (typeof part === 'string') return part;
                    const maybeObj = part as { text?: unknown };
                    return typeof maybeObj?.text === 'string' ? maybeObj.text : '';
                })
                .join('\n')
                .trim();
            }
            const maybeObj = c as { text?: unknown };
            if (maybeObj && typeof maybeObj.text === 'string') return maybeObj.text;
            try { return JSON.stringify(c); } catch { return ''; }
            };

            const normalized = items.map<Message>((m) => {
            const sender =
                (typeof m.sender === 'string' && m.sender) ||
                (typeof m.agent === 'string' && m.agent) ||
                (m.role === 'user' ? (userName || 'You') : (typeof m.model === 'string' ? m.model : 'Assistant'));

            const model =
                (typeof m.model === 'string' && m.model) ||
                (typeof m.sender === 'string' && m.sender) ||
                'Assistant';

            const text = toText(m.content ?? m.text ?? m.message ?? m.body ?? '');
            return { sender, model, text };
            });

            setChatHistory(prev => (prev.length === 0 ? normalized : prev));
        } catch {
            /* noop */
        }
        }, [userName]);

    useEffect(() => {
        // If there's an active conversation ID and the chat history is empty,
        // it's a signal that we need to hydrate the chat from the backend.
        if (conversationId && chatHistory.length === 0) {
            hydrateConversation(conversationId);
        }
    }, [conversationId, chatHistory.length, hydrateConversation]);
    // Fetch the list of conversations
const loadConversations = useCallback(async () => {
    try {
        setIsLoadingConvs(true);

        // Obtain a token from localStorage or userToken
        const token =
            typeof window !== 'undefined'
                ? localStorage.getItem('access_token') || userToken || null
                : null;

        // If no token, abort; prevents clearing the sidebar
        if (!token) return;

        // Build query string with limit and token
        const query = new URLSearchParams();
        query.set('limit', '100');
        query.set('token', token);

        const res = await apiFetch(`/api/conversations/by_token?${query.toString()}`, {
            cache: 'no-store',
        });
        if (!res.ok) throw new Error(`List convos failed: ${res.status}`);
        const data: ConversationListResponse = await res.json();

        // Flatten various response shapes
        const raw: ConversationListItem[] =
            Array.isArray(data)
                ? data
                : Array.isArray((data as { items?: ConversationListItem[] })?.items)
                ? (data as { items: ConversationListItem[] }).items
                : Array.isArray((data as { conversations?: ConversationListItem[] })?.conversations)
                ? (data as { conversations: ConversationListItem[] }).conversations
                : Array.isArray((data as { data?: ConversationListItem[] })?.data)
                ? (data as { data: ConversationListItem[] }).data
                : Array.isArray(
                    (data as { data?: { items?: ConversationListItem[] } })?.data?.items,
                  )
                ? ((data as { data: { items: ConversationListItem[] } }).data.items)
                : [];

        // Normalize & sort by updated_at or created_at
        const normalized: ConversationListItem[] = raw
            .map((c) => ({
                id: String(c.id),
                title: (c.title && String(c.title)) || 'Untitled',
                updated_at:
                    c.updated_at ??
                    (c as { updatedAt?: string }).updatedAt ??
                    (c as { last_updated?: string }).last_updated,
                created_at:
                    c.created_at ?? (c as { createdAt?: string }).createdAt ?? null,
            }))
            .filter((c) => !!c.id)
            .sort((a, b) => {
                const ta = Date.parse(String(a.updated_at ?? a.created_at ?? ''));
                const tb = Date.parse(String(b.updated_at ?? b.created_at ?? ''));
                return (tb || 0) - (ta || 0);
            });

        // De-duplicate by id
        const seen: Record<string, true> = {};
        const unique = normalized.filter((c) =>
            seen[c.id] ? false : (seen[c.id] = true),
        );
        setConversations(unique);
    } catch {
        // ignore errors
    } finally {
        setIsLoadingConvs(false);
    }
}, [userToken]);





    useEffect(() => {
        if (userToken) {
            loadConversations();
        }
    }, [userToken, loadConversations]);

    // Also try once on mount using whatever token is already in localStorage.
    // Covers refreshes where context isn't ready yet.
    useEffect(() => {
        loadConversations();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleOpenConversation = async (id: string) => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
        ws.current.close(1000, 'switch-conversation');
    }

    setConversationId(id);
    try { localStorage.setItem('conversationId', id); } catch {}

    setLoadedSummary(null);
    setShowSummary(false);
    setChatHistory([]);
    setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });

    hydrateConversation(id);
    };

    // Restore last-opened conversation ID from localStorage (before WS connects)
    useEffect(() => {
    try {
        const cid = localStorage.getItem('conversationId');
        if (cid) setConversationId(cid);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Use useCallback to memoize the function, preventing unnecessary re-renders
    const addMessageToChat = useCallback((msg: { sender: string; text: string }) => {
        const cleanSender = (msg.sender || '').trim();
        setChatHistory(prev => {
            const next = [
                ...prev,
                { sender: cleanSender, model: cleanSender, text: msg.text }
            ];
            // Prevent hydrate from firing based on a stale length in racey WS events
            chatLengthRef.current = next.length;
            return next;
        });
        // keep the newest message in view
        setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 0);
    }, []);

    


    useEffect(() => {
    const onVisible = () => {
        if (document.visibilityState === 'visible' && !ws.current) {
        // nudge a reconnect attempt
        setWsReconnectNonce((n) => n + 1);
        }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
    }, []);




    const handleSubmit = useCallback((e: React.FormEvent) => {
        e.preventDefault();

        Object.values(typingShowDelayRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
        Object.values(typingTTLRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
        typingShowDelayRef.current = {};
        typingTTLRef.current = {};
        setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });

        const text = message.trim();
        if (!text && pendingFiles.length === 0) return; // Allow sending only files
        
        if (!userName) {
            // Handle the case where the user is not logged in.
            // You might want to display an error or redirect.
            console.error("User name is missing, cannot send message.");
            return;
        }
        const payload: Record<string, unknown> = {
            message: text,
            attachments: pendingFiles.map(f => f.id)
        };

        try {
            const sock = ws.current;
            const open = sock && sock.readyState === WebSocket.OPEN;

            if (open) {
                sock!.send(JSON.stringify(payload));
            } else {
            // queue until socket is open
            pendingSends.current.push(payload);
            // ensure a reconnect attempt is queued if it’s closed
                if (!sock || sock.readyState === WebSocket.CLOSED) {
                    setWsReconnectNonce((n) => n + 1);
                }
            }

            // optimistic UI
            addMessageToChat({ sender: userName, text });
            setMessage('');
            setPendingFiles([]); // Clear pending files after sending

            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto'; // collapse back to 1 line
            }
        } catch (err) {
            console.error('Send failed:', err);
        }
    }, [message, userName, addMessageToChat, pendingFiles]);

    // Create a brand-new conversation
    const handleNewConversation = () => {  
        setConversationId(null);
        try { localStorage.removeItem('conversationId'); } catch {}
        setLoadedSummary(null);
        setShowSummary(false);
        setChatHistory([]);
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            ws.current.close(1000, 'new-conversation');
        }
        ws.current = null;
        // The WebSocket reconnect effect will fire automatically because conversationId changed to null.
        // We do not need a manual timeout or nonce bump here.
    };


    // Rename the selected conversation
    const handleRenameConversation = async (id: string) => {
        if (!userToken) return;

        const current = conversations.find((c) => c.id === id);
        const proposed = window.prompt('Rename conversation to:', current?.title ?? '');
        if (!proposed || !proposed.trim()) return;
        const h = buildAuthHeaders(userToken);
        h.set("Content-Type", "application/json");
        const res = await apiFetch(`/api/conversations/${id}`, {
            method: 'PATCH',
            headers: h,
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

        const res = await apiFetch(`/api/conversations/${id}`, {         
            method: "DELETE",
            headers: buildAuthHeaders(userToken),  // <— add this 
        });
        if (!res.ok) {
            alert('Delete failed.');
            return;
        }

        if (conversationId === id) {
            try { localStorage.removeItem('conversationId'); } catch {}
            handleNewConversation();
        } else {
            await loadConversations();
        }
    };

    const toggleSelect = useCallback((id: string) => {
    setSelectedIds(prev => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
    });
    }, []);

    const selectAll = useCallback(() => {
    setSelectedIds(new Set(conversations.map(c => c.id)));
    }, [conversations]);

    const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    }, []);

    const handleBulkDelete = useCallback(async () => {
    if (!userToken) return;
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;

    if (!window.confirm(`Delete ${ids.length} conversation${ids.length > 1 ? 's' : ''}? This cannot be undone.`)) {
        return;
    }

    // optimistic removal in UI
    setConversations(prev => prev.filter(c => !selectedIds.has(c.id)));

    await Promise.allSettled(
        ids.map(id =>
        apiFetch(`/api/conversations/${id}`, {
            method: 'DELETE',
            headers: buildAuthHeaders(userToken),
        })
        )
    );

    // if the open convo got deleted, reset; otherwise refresh list
    const deletedOpen = conversationId ? selectedIds.has(conversationId) : false;
    setSelectedIds(new Set());
    setManageMode(false);

    if (deletedOpen) {
        try { localStorage.removeItem('conversationId'); } catch {}
        handleNewConversation();
    } else {
        await loadConversations();
    }
    }, [userToken, selectedIds, conversationId, loadConversations, handleNewConversation, setConversations]);



    // eslint-disable-next-line react-hooks/exhaustive-deps
    // WebSocket connection logic
    useEffect(() => {
        let currentWs: WebSocket | null = null;
        (async () => {
        // Don't try to connect without a token.
        if (!userToken) {
            if (ws.current) {
                try { ws.current.close(); } catch {}
            }
            ws.current = null;
            setIsWsOpen(false);
            return;
        }
        
        // Try to top up access token if it's close to expiring (non-blocking).
        // Ensure we actually have a valid access token before connecting
        let token: string | null = null;
        try {
            token = await getFreshAccessToken();
        } catch {
            token = null; // continue anonymously
        }
        // proceed even if token is null


        const base = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
        let u: URL;
        try { u = new URL(base); }
        catch (e) {
        console.error("Invalid WS URL from env, falling back:", e);
        u = new URL('http://localhost:8000');
        }

        u.protocol = (u.protocol === 'https:' || u.protocol === 'wss:') ? 'wss:' : 'ws:';
        u.pathname = '/ws/colosseum-chat';
        if (token) {
            u.searchParams.set('token', token);
        }

        const socket = new WebSocket(u.toString());
        currentWs = socket;

        ws.current = socket;
        setIsWsOpen(false);
        
        const isNonEmptyString = (v: unknown): v is string =>
            typeof v === 'string' && v.length > 0;

        Object.assign(currentWs, {
            onopen: () => {
                reconnectRef.current.tries = 0;
                if (reconnectRef.current.timer) {
                    window.clearTimeout(reconnectRef.current.timer);
                    reconnectRef.current.timer = null;
                }
        
                setIsWsOpen(true);
                loadConversations();
                authFailedRef.current = false;
          
        
                const initialPayload: any = {
                    kind: 'start',
                    user_name: user?.first_name || user?.email || 'Guest'
                };
                if (conversationId) {
                    initialPayload.conversation_id = conversationId;
                }
                currentWs!.send(JSON.stringify(initialPayload));
                
                while (pendingSends.current.length > 0) {
                    const next = pendingSends.current.shift();
                    if (next) socket.send(JSON.stringify(next));
                }
            },
        
            onmessage: (event: MessageEvent<string>) => {
                let msg: ServerMessage;
                try { msg = JSON.parse(event.data); }
                catch { return; }

                if (msg.type === 'conversation_id' && typeof msg.id === 'string') {
                    const id = msg.id;
                    setConversationId(curr => curr || id);
                    try { localStorage.setItem('conversationId', id); } catch {}

                    setConversations(prev => {
                        const rest = prev.filter(c => c.id !== id);
                        return [{ id, title: 'New conversation', updated_at: new Date().toISOString() }, ...rest];
                    });
                    if (chatLengthRef.current === 0) { hydrateConversation(id); }
                    loadConversations();
                    return;
                }

                if (msg.type === 'conversation_meta' && typeof msg.id === 'string') {
                    const id: string = msg.id;
                    const title: string = typeof msg.title === 'string' && msg.title.trim() ? msg.title : 'New conversation';
                    const updated_at: string = typeof msg.updated_at === 'string' && msg.updated_at.trim() ? msg.updated_at : new Date().toISOString();
                    setConversations(prev => {
                        const rest = prev.filter(c => c.id !== id);
                        return [{ id, title, updated_at }, ...rest];
                    });
                    setConversationId(curr => {
                        const chosen = curr || id;
                        try { localStorage.setItem('conversationId', chosen); } catch {}

                        if (chatLengthRef.current === 0) { hydrateConversation(chosen); }
                        return chosen;
                    });
                    return;
                }

                if (msg.type === 'context_summary' && typeof msg.summary === 'string' && msg.summary.trim()) {
                    setLoadedSummary(msg.summary);
                    return;
                }
                if (msg.type === 'ping' || msg.type === 'pong') return;
                const sender = isNonEmptyString(msg.sender) ? msg.sender : null;

                if (sender && typeof msg.typing === 'boolean' && ALLOWED_AGENTS.includes(sender as AgentName)) {
                    setTypingWithDelayAndTTL(sender as AgentName, msg.typing === true);
                    return;
                }

                if (sender && typeof msg.text === 'string' && ALLOWED_AGENTS.includes(sender as AgentName)) {
                    setTypingWithDelayAndTTL(sender as AgentName, false);
                    addMessageToChat({ sender, text: msg.text });
                    return;
                }
            },
            onclose: (ev: CloseEvent) => {
                console.log('WebSocket closed. code=', ev.code, 'reason=', ev.reason, 'wasClean=', ev.wasClean);
                Object.values(typingTimersRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                typingTimersRef.current = {};
                Object.values(typingShowDelayRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                Object.values(typingTTLRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                typingShowDelayRef.current = {};
                typingTTLRef.current = {};
                setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });
                setIsWsOpen(false);
                ws.current = null;
                const base = 1000;
                const max = 15000;
                const tries = reconnectRef.current.tries;
                const nextDelay = Math.min(max, base * Math.pow(2, tries)) + Math.floor(Math.random() * 250);
                reconnectRef.current.tries = tries + 1;
                if (reconnectRef.current.timer) {
                    window.clearTimeout(reconnectRef.current.timer);
                }
                reconnectRef.current.timer = window.setTimeout(() => {
                    setWsReconnectNonce((n) => n + 1);
                }, nextDelay);
            },
            onerror: (event: Event) => {
                console.error('WebSocket error:', event);
                Object.values(typingTimersRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                typingTimersRef.current = {};                
                setIsWsOpen(false);
                Object.values(typingShowDelayRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                Object.values(typingTTLRef.current).forEach(id => typeof id === 'number' && window.clearTimeout(id));
                typingShowDelayRef.current = {};
                typingTTLRef.current = {};
                setIsTyping({ ChatGPT:false, Claude:false, Gemini:false, Mistral:false });
            }
        });
        })();
        return () => {
            if (
            currentWs &&
            (currentWs.readyState === WebSocket.OPEN || currentWs.readyState === WebSocket.CONNECTING)
            ) {
            try { currentWs.close(); } catch {}
            }
        };        
        }, [userToken, userName, addMessageToChat, wsReconnectNonce, conversationId, loadConversations, hydrateConversation]);


        

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
        <aside className="hidden md:flex flex-col w-72 border-r border-gray-800 bg-gray-900 pt-[72px] z-[60]">

        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div className="font-semibold">Conversations</div>
        <div className="flex items-center gap-2">
            <button
            onClick={() => setManageMode(m => !m)}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
            title={manageMode ? "Done" : "Manage conversations"}
            >
            {manageMode ? 'Done' : 'Manage'}
            </button>
            <button
            onClick={handleNewConversation}
            className="text-xs px-2 py-1 rounded bg-blue-600 hover:bg-blue-700"
            >
            New
            </button>
        </div>
        </div>

        {manageMode && (
        <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-2">
            <button
            onClick={selectAll}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
            >
            Select all
            </button>
            <button
            onClick={clearSelection}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
            >
            Clear
            </button>
            <button
            onClick={handleBulkDelete}
            disabled={selectedIds.size === 0}
            className="text-xs px-2 py-1 rounded bg-red-700 hover:bg-red-600 disabled:opacity-50"
            title={selectedIds.size === 0 ? "No conversations selected" : "Delete selected"}
            >
            Delete selected
            </button>
            <span className="ml-auto text-[11px] text-gray-400">{selectedIds.size} selected</span>
        </div>
        )}


        <div className="flex-1 min-w-0 overflow-y-auto custom-scrollbar">
            {isLoadingConvs && (
            <div className="p-4 text-sm text-gray-400">Loading…</div>
            )}
            {!isLoadingConvs && conversations.length === 0 && (
            <div className="p-4 text-sm text-gray-400">No conversations yet</div>
            )}
            <ul className="w-full divide-y divide-gray-800">
                {conversations.map((c) => (
                    <li
                        key={c.id}
                        onClick={() => manageMode ? toggleSelect(c.id) : handleOpenConversation(c.id)}
                        className={`group relative w-full px-3 py-2 cursor-pointer ${
                            conversationId === c.id ? 'bg-gray-800' : 'hover:bg-gray-800/60'
                        }`}
                    >

                        <div className="flex items-center justify-between gap-2">
                            <div
                                className={`flex items-center gap-2 min-w-0 ${
                                    manageMode
                                    ? 'pr-2'
                                    : 'pr-6 group-hover:pr-20 md:group-hover:pr-24 transition-[padding-right] duration-200'
                                }`}
                            >

                            {manageMode && (
                                <input
                                type="checkbox"
                                className="h-4 w-4"
                                checked={selectedIds.has(c.id)}
                                onChange={(e) => { e.stopPropagation(); toggleSelect(c.id); }}
                                onClick={(e) => e.stopPropagation()}
                                />
                            )}
                            <span className="flex-1 text-left break-words whitespace-normal leading-snug" title={c.title}>
                                {c.title || 'Untitled'}
                            </span>
                            </div>

                            {/* Hide per-item actions in manage mode */}
                            <div
                                className={`${
                                    manageMode
                                    ? 'hidden'
                                    : 'opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto'
                                } absolute right-3 top-2 w-16 sm:w-20 transition-opacity flex flex-col gap-1 items-stretch`}
                            >

                                <button
                                    onClick={(e) => { e.stopPropagation(); handleRenameConversation(c.id); }}
                                    className="w-full text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
                                    title="Rename"
                                >
                                    Rename
                                </button>
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleDeleteConversation(c.id); }}
                                    className="w-full text-xs px-2 py-1 rounded bg-red-700 hover:bg-red-600"
                                    title="Delete"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>

                        {(c.updated_at || c.created_at) && (
                            <div className="mt-1 text-[11px] text-gray-400">
                            {new Date((c.updated_at ?? c.created_at) as string).toLocaleString()}
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
            className="flex-1 overflow-y-auto pt-[72px] bg-gray-900 custom-scrollbar"
            style={{ paddingBottom: composerHeight + 24 }}  // extra 24px breathing room
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
                chatHistory.map((msg, index) => {
                    const isUser =
                    (msg.sender || '').trim().toLowerCase() === (userName || 'You').trim().toLowerCase();
                    return (
                    <div key={index} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                        <div
                        className={`relative p-3 md:p-4 max-w-[80%] text-white rounded-2xl md:rounded-3xl shadow-lg transition-all duration-200 ease-in-out transform hover:scale-[1.01] ${
                            isUser
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
                        {!isUser && (
                            <div className="text-xs text-gray-300 mb-1 font-semibold">{msg.sender}</div>
                        )}
                        <pre className="whitespace-pre-wrap font-sans text-sm md:text-base leading-relaxed">
                            {msg.text}
                        </pre>
                        </div>
                    </div>
                    );
                })}


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
{/* ===== Composer (fixed footer) ===== */}
<div
    ref={composerRef}
    className="fixed bottom-0 left-0 right-0 md:left-72 z-50 bg-gray-900 border-t border-gray-800"
>
  {/* Summary banner (if you have one) can live above the form if you like */}
  {loadedSummary && (
    <div className="max-w-4xl mx-auto px-4 pt-3">
      <div className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
        <div className="flex items-start justify-between gap-3">
          <div>
            <strong>Context loaded</strong> — this chat includes knowledge from earlier sessions.
          </div>
          <button
            type="button"
            className="shrink-0 underline"
            onClick={() => setShowSummary(s => !s)}
          >
            {showSummary ? "Hide" : "Show"} summary
          </button>
        </div>
        {showSummary && (
          <div className="mt-2 whitespace-pre-wrap">{loadedSummary}</div>
        )}
      </div>
    </div>
  )}

  <form onSubmit={handleSubmit} className="max-w-4xl mx-auto p-4 md:p-6">
    <div className="flex gap-4">
        <div className="flex flex-col items-start gap-2 pt-1">
            <label className="inline-flex items-center px-3 py-2 rounded-xl bg-gray-800 border border-gray-700 text-sm cursor-pointer hover:bg-gray-700">
                <input
                type="file"
                accept=".txt,.md,.json,.js,.ts,.tsx,.jsx,.py,.java,.go,.rb,.php,.css,.html,.sql,.yaml,.yml,.sh,.c,.cpp,.cs,.rs,.kt"
                multiple
                onChange={handleFilePick}
                className="hidden"
                />
                Attach files
            </label>

            {/* chips for pending files */}
            {pendingFiles.length > 0 && (
                <div className="flex flex-wrap gap-2">
                {pendingFiles.map(f => (
                    <span key={f.id} className="inline-flex items-center gap-2 px-2 py-1 text-xs rounded-full bg-gray-800 border border-gray-700">
                    <span className="truncate max-w-[180px]" title={f.name}>{f.name}</span>
                    <button type="button" onClick={() => removePending(f.id)} className="text-gray-300 hover:text-white">×</button>
                    </span>
                ))}
                </div>
            )}
        </div>

        <textarea
            ref={textareaRef}
            rows={1}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onInput={(e) => {
                const ta = e.currentTarget;
                ta.style.height = 'auto';               // shrink back down if needed
                ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'; // cap ~5 lines
            }}
            onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                (e.currentTarget.form as HTMLFormElement)?.requestSubmit();
                }
            }}
            placeholder="Type your message..."
            className="flex-grow p-3 rounded-xl border border-gray-700 bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 text-sm md:text-base resize-none overflow-y-auto min-h-[44px] max-h-40"
            disabled={!userName}
        />      
        <button
            type="submit"
            className="h-11 md:h-12 px-4 md:px-6 self-end bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-colors duration-200 text-sm"
            disabled={!userName}
        >
            Send
        </button>
    </div>
  </form>
</div>
{/* ===== /Composer ===== */}

        </div>
        <style jsx global>{`
        /* Works on Firefox */
        .custom-scrollbar {
            scrollbar-width: thin;
            scrollbar-color: #475569 #0f172a; /* thumb, track */
        }

        /* Works on Chrome, Edge, Safari */
        .custom-scrollbar::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #0f172a; /* dark track to match your theme */
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #475569;        /* slate-600 */
            border-radius: 9999px;       /* fully rounded */
            border: 2px solid #0f172a;   /* creates a gap around the thumb */
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #64748b; /* slate-500 on hover */
        }
        `}</style>

    </div>
    );

}