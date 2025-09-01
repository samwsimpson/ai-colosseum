'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import type React from 'react';
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
type UploadedAttachment = {
  id: string;
  name: string;
  mime: string;
  size?: number | null;
  signed_url: string;     // NOTE: snake_case in state for consistency
  signedUrl?: string;     // legacy camelCase (optional)
  content?: string;       // optional preview for text files
};


type ConversationListResponse =
  | ConversationListItem[]
  | { items: ConversationListItem[] }
  | { conversations: ConversationListItem[] }
  | { data: ConversationListItem[] }
  | { data: { items: ConversationListItem[] } };

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
        ? (pathOrUrl.startsWith('http') ? pathOrUrl : `${API_BASE}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`)
        : pathOrUrl.toString();

    // Use a temporary headers object to avoid modifying the original
    const finalHeaders = new Headers(init.headers || {});
    
    // Add the Authorization header
    const access = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    if (access) finalHeaders.set('Authorization', `Bearer ${access}`);

    // Critically, do not set Content-Type if a FormData body is provided
    if (init.body instanceof FormData) {
        finalHeaders.delete('Content-Type');
    } else if (!finalHeaders.has('Content-Type')) {
        // Set the default only if it's not FormData and not already set
        finalHeaders.set('Content-Type', 'application/json');
    }

    let res = await fetch(url, { ...init, headers: finalHeaders, credentials: 'include' });
    if (res.status !== 401) return res;
    
    // Retry logic
    const rr = await fetch(`${API_BASE}/api/refresh`, { method: 'POST', credentials: 'include' });
    if (!rr.ok) return res;

    const { token } = await rr.json();
    if (token) {
        if (typeof window !== 'undefined') localStorage.setItem('access_token', token);
        finalHeaders.set('Authorization', `Bearer ${token}`);
        res = await fetch(url, { ...init, headers: finalHeaders, credentials: 'include' });
    }
    return res;
}


type AgentName = 'ChatGPT' | 'Claude' | 'Gemini' | 'Mistral';
const ALLOWED_AGENTS: AgentName[] = ['ChatGPT', 'Claude', 'Gemini', 'Mistral'];
interface UploadListItem {
  id: string;
  name: string;
  mime: string;
  size?: number | null;
  signed_url?: string;
  created_at?: string | null;
  ts?: number;                      // numeric timestamp for sorting
  from: 'user' | 'assistant';
  agent?: AgentName;
}

function uid() {
    return Math.random().toString(36).substring(2, 10);
}
export default function ChatPage() {

    interface TypingTimers { showDelay: number | null; ttl: number | null; }
    const typingTimersRef = useRef<Partial<Record<AgentName, TypingTimers>>>({});

    /** Only show bubble if the agent is still "typing" after showDelayMs.
     * When shown, auto-clear after ttlMs unless a message/false arrives. */
    const setTypingWithDelayAndTTL = useCallback(
    (agent: AgentName, value: boolean, showDelayMs = 400, ttlMs = 12000) => {
        const timers = typingTimersRef.current[agent] ?? { showDelay: null, ttl: null };

        if (timers.showDelay) window.clearTimeout(timers.showDelay);
        if (timers.ttl) window.clearTimeout(timers.ttl);

        if (!value) {
        setIsTyping(prev => ({ ...prev, [agent]: false }));
        delete typingTimersRef.current[agent];
        if (!authChecked) return null;  // or a skeleton loader if you prefer
        if (!authed) return null;       // redirect already in flight

        return;
        }

        timers.showDelay = window.setTimeout(() => {
        setIsTyping(prev => ({ ...prev, [agent]: true }));

        if (timers.ttl) window.clearTimeout(timers.ttl);
        timers.ttl = window.setTimeout(() => {
            setIsTyping(prev => ({ ...prev, [agent]: false }));
            if (typingTimersRef.current[agent]) delete typingTimersRef.current[agent];
        }, ttlMs);
        }, showDelayMs);

        typingTimersRef.current[agent] = timers;
    },
    []
    );

const { userName, userToken } = useUser();
const router = useRouter();
const pathname = usePathname();
const [authChecked, setAuthChecked] = useState(false);
const [authed, setAuthed] = useState(false);


// === State for the UI and chat logic ===
const [message, setMessage] = useState<string>('');
const [chatHistory, setChatHistory] = useState<Message[]>([]);
const [uploadsList, setUploadsList] = useState<UploadListItem[]>([]);
const [isWsOpen, setIsWsOpen] = useState<boolean>(false);
const [loadedSummary, setLoadedSummary] = useState<string | null>(null);
const [showSummary, setShowSummary] = useState(false);
const [wsReconnectNonce, setWsReconnectNonce] = useState(0);
const [conversationId, setConversationId] = useState<string | null>(null);
const [pendingFiles, setPendingFiles] = useState<UploadedAttachment[]>([]);
const [, setIsUploading] = useState<boolean>(false);
const [conversations, setConversations] = useState<ConversationListItem[]>([]);
const [isLoadingConvs, setIsLoadingConvs] = useState(false);
const [manageMode, setManageMode] = useState(false);
const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
const [composerHeight, setComposerHeight] = useState<number>(120);
const [isTyping, setIsTyping] = useState<TypingState>({
    ChatGPT: false,
    Claude: false,
    Gemini: false,
    Mistral: false,
});

// NEW: global fallback typing indicator
const [teamThinking, setTeamThinking] = useState(false);
const teamThinkingHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

// helper to safely stop the fallback
const stopTeamThinking = useCallback(() => {
  if (teamThinkingHideTimer.current) {
    clearTimeout(teamThinkingHideTimer.current);
    teamThinkingHideTimer.current = null;
  }
  setTeamThinking(false);
}, []);

// Extract any upload/file metadata from a raw message payload (server formats vary)
const extractUploadsFromRawMessage = useCallback((raw: unknown): UploadListItem[] => {
  const out: UploadListItem[] = [];
  if (!raw || typeof raw !== "object") return out;

  const r = raw as Record<string, unknown>;
  const senderStr = typeof r.sender === "string" ? r.sender : undefined;
  const agent = senderStr && (ALLOWED_AGENTS as string[]).includes(senderStr) ? (senderStr as AgentName) : undefined;
  const from: "user" | "assistant" = agent ? "assistant" : (r.role === "user" ? "user" : "assistant");

  const arrays: unknown[] = [];
  const take = (v: unknown) => { if (Array.isArray(v)) arrays.push(...v); };

  take(r.file_metadata_list);
  take(r.files);
  take(r.attachments);
  take(r.uploads);
  take(r.file_metadata);

  const inferNameFromUrl = (s: string): string => {
    try {
      const url = new URL(s, typeof window !== "undefined" ? window.location.href : "https://example.com/");
      const last = url.pathname.split("/").pop();
      return last && last.trim() ? last : "file";
    } catch {
      const parts = s.split("/");
      return parts[parts.length - 1] || "file";
    }
  };

  const toItem = (v: unknown): UploadListItem | null => {
    if (!v || typeof v !== "object") return null;
    const o = v as Record<string, unknown>;

    const signed =
      (typeof o.signed_url === "string" && o.signed_url) ||
      (typeof o.url === "string" && o.url) ||
      undefined;

    const name =
      (typeof o.name === "string" && o.name) ||
      (typeof o.filename === "string" && o.filename) ||
      (signed ? inferNameFromUrl(signed) : "file");

    const mime =
      (typeof o.mime === "string" && o.mime) ||
      (typeof o.content_type === "string" && o.content_type) ||
      "application/octet-stream";

    const size = typeof o.size === "number" ? o.size : null;
    const created_at = typeof o.created_at === "string" ? o.created_at : null;
    const id =
      (typeof o.id === "string" && o.id) ||
      signed ||
      name ||
      crypto.randomUUID();

    const ts = created_at ? Date.parse(created_at) : Date.now();
    return { id, name, mime, size, signed_url: signed, created_at, ts, from, agent };
  };

  for (const v of arrays) {
    const i = toItem(v);
    if (i) out.push(i);
  }
  return out;
}, []);

// === Refs for DOM elements and internal state ===
const chatContainerRef = useRef<HTMLDivElement | null>(null);
const composerRef = useRef<HTMLDivElement | null>(null);
const textareaRef = useRef<HTMLTextAreaElement | null>(null);
const isComposingRef = useRef(false);
const isInitialRender = useRef(true);
const chatEndRef = useRef<HTMLDivElement>(null);
const ws = useRef<WebSocket | null>(null);
const chatLengthRef = useRef(0);
const pendingSends = useRef<Array<Record<string, unknown>>>([]);
const isSubmittingRef = useRef<boolean>(false);
const lastUserTextRef = useRef<string | null>(null);
const lastUserClientIdRef = useRef<string | null>(null);
const reconnectRef = useRef<{ tries: number; timer: number | null }>({
    tries: 0,
    timer: null,
});
const authFailedRef = useRef(false);
const reconnectBackoffRef = useRef(1000); // start at 1s, exponential up to 15s
const refreshInFlight = useRef<Promise<void> | null>(null);
const lastRefreshAt = useRef<number>(0);
const fileInputRef = useRef<HTMLInputElement>(null);
// Normalize a wide variety of LLM message payloads into a plain string
function normalizeToPlainText(input: unknown): string {
  if (input == null) return '';

  if (typeof input === 'object') {
    const obj = input as Record<string, unknown>;

    if (typeof obj.text === 'string') return obj.text;
    if (typeof obj.message === 'string') return obj.message;

    const content = obj.content as unknown;
    if (typeof content === 'string') return content;

    if (Array.isArray(content)) {
      const parts = content
        .map((p: unknown) => {
          if (typeof p === 'string') return p;

          const asText = p as { text?: unknown };
          if (asText && typeof asText.text === 'string') return asText.text;

          const asTyped = p as { type?: unknown; text?: unknown };
          if (
            asTyped &&
            typeof asTyped.type === 'string' &&
            asTyped.type === 'text' &&
            typeof asTyped.text === 'string'
          ) {
            return asTyped.text;
          }
          return '';
        })
        .filter(Boolean);
      return parts.join(' ').trim();
    }

    if (typeof obj.body === 'string') return obj.body;
  }

  if (typeof input === 'string') return input;

  return '';
}


// 1) single-responsibility upload helper: returns the uploaded attachment but DOES NOT set state

const uploadOne = async (file: File): Promise<UploadedAttachment> => {
  const form = new FormData();
  form.append("file", file);

  const res = await apiFetch("/api/upload-file", { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);

  const raw = await res.json() as {
    id?: string;
    name?: string;
    mime?: string;
    size?: number | null;
    signed_url?: string;
    filename?: string;     // legacy
    url?: string;          // legacy
    content?: string;
    content_type?: string; // legacy
  };

  return {
    id: raw.id ?? raw.url ?? crypto.randomUUID(),
    name: raw.name ?? raw.filename ?? file.name,
    mime: raw.mime ?? raw.content_type ?? file.type ?? "application/octet-stream",
    size: raw.size ?? file.size,
    signed_url: raw.signed_url ?? raw.url!,  // normalize to signed_url
    content: raw.content,
  };
};

async function handleFilePick(e: React.ChangeEvent<HTMLInputElement>) {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;

  setIsUploading(true);
  try {
    const uploaded: UploadedAttachment[] = [];
    for (const f of files) uploaded.push(await uploadOne(f));
    // FIX: correct spread (no ".prev" typo)
    setPendingFiles(prev => [...prev, ...uploaded]);
  } catch (err) {
    console.error(err);
    alert("Upload failed. Try smaller files.");
  } finally {
    setIsUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }
}


const removePending = (id: string) => setPendingFiles(prev => prev.filter(p => p.id !== id));

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

    // Sidebar bulk-manage state    
    
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

            // Seed the uploads list from any historical messages that contain files
            const seedUploads: UploadListItem[] = [];
            for (const m of items) {
            const got = extractUploadsFromRawMessage(m);
            if (got.length) seedUploads.push(...got);
            }
            if (seedUploads.length) {
                setUploadsList(seedUploads.sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0)));
            }


            setChatHistory(prev => (prev.length === 0 ? normalized : prev));
        } catch {
            /* noop */
        }
        }, [userName, userToken, extractUploadsFromRawMessage]);




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
        const raw: unknown[] =
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
        type Loose = Record<string, unknown>;

        const parseDateMs = (v: unknown): number => {
        if (v instanceof Date) return v.getTime();
        if (typeof v === 'number') return Number.isFinite(v) ? v : 0;
        if (typeof v === 'string') {
            const t = Date.parse(v);
            return Number.isFinite(t) ? t : 0;
        }
        return 0;
        };

        const pickTimestampMs = (o: Loose): number => {
        return parseDateMs(
            o['updated_at'] ?? o['updatedAt'] ?? o['last_updated'] ?? o['created_at'] ?? o['createdAt']
        );
        };

        const getStr = (o: Loose, k: string): string | null =>
        typeof o[k] === 'string' ? (o[k] as string) : null;

        const normalized: ConversationListItem[] = (raw as Loose[])
            .filter((c) => c?.id != null)
            .sort((a, b) => pickTimestampMs(b) - pickTimestampMs(a))
            .map((c) => {
                const updated =
                getStr(c, 'updated_at') ??
                getStr(c, 'updatedAt') ??
                getStr(c, 'last_updated') ??
                undefined; // coerce null -> undefined

                const created =
                getStr(c, 'created_at') ??
                getStr(c, 'createdAt') ??
                null; // allow null

                return {
                id: String(c.id),
                title: (typeof c.title === 'string' && c.title) ? c.title : 'Untitled',
                updated_at: updated,
                created_at: created,
            };
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
    setUploadsList([]);
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

    useEffect(() => {
        // Whenever we switch conversations (including when the server assigns a new one),
        // start the uploads list empty; hydrateConversation will refill it.
        setUploadsList([]);
    }, [conversationId]);

    // Auth gate: never render chat unless authenticated (no UI flash)
    useEffect(() => {
    let cancelled = false;

    const checkAuthAndLoad = async () => {
        try {
        // 1) Context token → authed
        if (userToken) {
            if (!cancelled) { setAuthed(true); setAuthChecked(true); }
            loadConversations();
            setWsReconnectNonce(n => n + 1);
            return;
        }

        // 2) Ask backend for a fresh access token if the HttpOnly refresh cookie exists
        //    (middleware ensures guests don't reach this page, but this covers race cases)
        const res = await fetch('/api/refresh', { method: 'POST', credentials: 'include' });
        if (res.ok) {
            const body = await res.json();
            try { localStorage.setItem('access_token', body.access_token); } catch {}
            if (!cancelled) { setAuthed(true); setAuthChecked(true); }
            loadConversations();
            setWsReconnectNonce(n => n + 1);
            return;
        }
        } catch {
        // ignore
        }

        // 3) Not authed → clear & redirect (no UI flash)
        try { localStorage.removeItem('access_token'); } catch {}
        setConversationId(null);
        setChatHistory([]);
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });

        if (ws.current && ws.current.readyState < 2) {
        ws.current.close(1000, 'logout');
        }
        ws.current = null;

        if (!cancelled) {
        setAuthed(false);
        setAuthChecked(true);
        }

        if (pathname !== '/sign-in') {
        router.replace(
            '/sign-in' + (pathname && pathname !== '/chat' ? `?next=${encodeURIComponent(pathname)}` : '')
        );
        }
    };

    checkAuthAndLoad();
    return () => { cancelled = true; };
    }, [userToken, pathname, router, loadConversations]);



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

    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        if (isSubmittingRef.current) return;
        isSubmittingRef.current = true;
        setTimeout(() => { isSubmittingRef.current = false; }, 300);

        e.preventDefault();

        const text = message.trim();
        const hasText = text.trim().length > 0;
        const hasFiles = pendingFiles.length > 0;
        if (!hasText && !hasFiles) return;

        Object.keys(typingTimersRef.current).forEach(k => {
            const t = typingTimersRef.current[k as AgentName];
            if (t?.showDelay) window.clearTimeout(t.showDelay);
            if (t?.ttl) window.clearTimeout(t.ttl);
        });
        typingTimersRef.current = {};
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });


        if (!userName) {
            console.error('User name is missing, cannot send message.');
            return;
        }

        const clientId = uid();
        lastUserClientIdRef.current = clientId;

        // Build the base message payload        
        const file_metadata_list = pendingFiles.map(f => ({
        id: f.id,
        filename: f.name,
        url: f.signed_url,
        content_type: f.mime || "application/octet-stream",
        size: typeof f.size === "number" ? f.size : undefined,
        content: f.content ?? undefined,  // text preview if available
        }));

        const placeholder = hasFiles && !hasText
            ? `(sent ${pendingFiles.length} file${pendingFiles.length > 1 ? "s" : ""})`
            : "";

        const payload: Record<string, unknown> = {
            client_id: clientId,
            user_name: userName,
            conversation_id: conversationId || undefined,
            attachments: pendingFiles.map(f => f.id),  // ids for server lookup
            text: hasText ? text : placeholder,        // send BOTH keys
            message: hasText ? text : placeholder,
            file_metadata_list,                        // the array you built above
            file_metadata: file_metadata_list[0],      // back-compat (ok if unused)
        };

        lastUserTextRef.current = text;
        // Show a global fallback immediately; hide automatically after 20s if nothing arrives.
        setTeamThinking(true);
        if (teamThinkingHideTimer.current) clearTimeout(teamThinkingHideTimer.current);
        teamThinkingHideTimer.current = setTimeout(() => setTeamThinking(false), 20000);

        try {
            const sock = ws.current;
            const open = sock && sock.readyState === WebSocket.OPEN;

            if (open) {
                sock!.send(JSON.stringify(payload));
            } else {
                pendingSends.current.push(payload);
                if (!sock || sock.readyState === WebSocket.CLOSED) {
                    setWsReconnectNonce(n => n + 1);
                }
            }

            addMessageToChat({ sender: userName, text: hasText ? text : placeholder });


            setMessage('');
            if (pendingFiles.length > 0) {
                const nowIso = new Date().toISOString();
                setUploadsList(prev => [
                    ...prev,
                    ...pendingFiles.map(f => ({
                    id: f.id,
                    name: f.name,
                    mime: f.mime,
                    size: f.size,
                    signed_url: f.signed_url,
                    created_at: nowIso,
                    ts: Date.now(),
                    from: 'user' as const,
                    })),
                ].sort((a, b) => (b.ts || 0) - (a.ts || 0)));
            }

            setPendingFiles([]);

            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        } catch (err) {
            console.error('Send failed:', err);
        }
    }, [message, userName, addMessageToChat, pendingFiles, conversationId]);

    // Create a brand-new conversation
    const handleNewConversation = useCallback(() => {  
        setConversationId(null);
        try { localStorage.removeItem('conversationId'); } catch {}
        setLoadedSummary(null);
        setShowSummary(false);
        setChatHistory([]);
        setUploadsList([]);  // ← clear uploads for a fresh convo
        setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });
        if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
            ws.current.close(1000, 'new-conversation');
        }
        ws.current = null;
        // The WebSocket reconnect effect will fire automatically because conversationId changed to null.
        // We do not need a manual timeout or nonce bump here.
    }, []);


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


    const resetWebSocket = useCallback(() => {
    // Clear all typing timers
    Object.keys(typingTimersRef.current).forEach(key => {
        const agent = key as AgentName;
        const timers = typingTimersRef.current[agent];
        if (!timers) return;
        if (timers.showDelay) window.clearTimeout(timers.showDelay);
        if (timers.ttl) window.clearTimeout(timers.ttl);
    });
    typingTimersRef.current = {};

    setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });
    setIsWsOpen(false);

    // Close and null socket (guarded)
    if (ws.current) {
        try {
        if (ws.current.readyState < 2) ws.current.close();
        } catch { /* ignore */ }
        finally { ws.current = null; }
    }
    }, []);

    // WebSocket connection logic
    useEffect(() => {
        // Only proceed if a userToken is present and valid.
        if (!userToken) {
            if (ws.current) {
                try { ws.current.close(); } catch {}
            }
            ws.current = null;
            setIsWsOpen(false);
            return;
        }

        // Build the URL safely
        const base = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
        let u: URL;
        try {
            u = new URL(base);
        } catch (e) {
            console.error("Invalid WS URL from env, falling back:", e);
            u = new URL('http://localhost:8000');
        }

        u.protocol = (u.protocol === 'https:' || u.protocol === 'wss:') ? 'wss:' : 'ws:';
        u.pathname = '/ws/colosseum-chat';
        
        let cleanup: (() => void) | null = null;

        // Use a function that gets the token, and if expired, refreshes it
        async function getTokenForWs() {
            const token = userToken;
            if (!token) {
                console.error("Attempted to connect WebSocket without a token.");
                return null;
            }

            // You'll need a JWT decoder library to check for expiration
            try {
                const decoded = JSON.parse(atob(token.split('.')[1] || ''));
                const now = Math.floor(Date.now() / 1000);
                if (decoded?.exp && decoded.exp < now) {
                    try {
                        const res = await apiFetch('/api/refresh', { method: 'POST' });
                        if (res.ok) {
                        const { token: newToken } = await res.json();
                        try { localStorage.setItem('access_token', newToken); } catch {}
                        return newToken;
                        }
                    } catch (e) {
                        console.error("Failed to refresh token for WebSocket", e);
                    }
                    // Hard stop: prevent any reconnect loop if we cannot refresh.
                    authFailedRef.current = true;
                    // Proactively clear any socket and mark closed.
                    if (ws.current && ws.current.readyState < 2) {
                        try { ws.current.close(1000, 'auth-refresh-failed'); } catch {}
                    }
                    ws.current = null;
                    setIsWsOpen(false);
                    // Navigate to sign-in once.
                    if (pathname !== '/sign-in') router.push('/sign-in');
                    return null;
                }

            } catch {
                // If we can't decode, still try the token — the server will enforce validity.
                return token;
            }
            return token;
        }

        getTokenForWs().then(tokenToSend => {
            if (!tokenToSend) {
                // If we couldn't get a valid token, don't connect.
                return;
            }
            u.search = `?token=${encodeURIComponent(tokenToSend)}`;
            const socket = new WebSocket(u.toString());
            ws.current = socket;
            setIsWsOpen(false);

            const currentWs = socket;            

            Object.assign(currentWs, {
                onopen: () => {
                    reconnectRef.current.tries = 0;
                    if (reconnectRef.current.timer) {
                        window.clearTimeout(reconnectRef.current.timer);
                    }
                    reconnectRef.current.timer = null;
            
                    setIsWsOpen(true);
                    loadConversations();
                    authFailedRef.current = false;
                    reconnectBackoffRef.current = 1000;
            
                    const initialPayload: Record<string, unknown> = {
                        user_name: userName,
                    };
                    if (conversationId) {
                        // Hydrate the conversation before connecting.
                        hydrateConversation(conversationId);
                        initialPayload.conversation_id = conversationId;
                    }
                    currentWs.send(JSON.stringify(initialPayload));
            
                    while (pendingSends.current.length > 0) {
                        const next = pendingSends.current.shift();
                        if (next) currentWs.send(JSON.stringify(next));
                    }
                },
            
                onmessage: (event: MessageEvent<string>) => {
                    let msg: ServerMessage;
                    try { msg = JSON.parse(event.data); }
                    catch { return; }

                    // Normalize a sender string we can trust for typing + message routing
                    const m = msg as Record<string, unknown>;
                    const sender =
                    (typeof m.sender === 'string' && m.sender.trim()) ? m.sender.trim()
                        : (typeof m.model === 'string' && m.model.trim()) ? m.model.trim()
                        : '';




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
                    if (sender && typeof msg.typing === 'boolean' && ALLOWED_AGENTS.includes(sender as AgentName)) {
                        // Any typing signal proves the pipeline is alive → hide the global fallback
                        stopTeamThinking();
                        setTypingWithDelayAndTTL(sender as AgentName, msg.typing === true);
                        return;
                    }

                    // Accept multiple server payload shapes, not just `text`
                    const unifiedText = normalizeToPlainText(
                        (m.text as unknown) ?? (m.content as unknown) ?? (m.message as unknown) ?? msg
                    );


                    if (sender && ALLOWED_AGENTS.includes(sender as AgentName) && unifiedText.trim().length > 0) {
                        // First real text → hide fallback and clear per-agent typing
                        stopTeamThinking();
                        setTypingWithDelayAndTTL(sender as AgentName, false);
                        addMessageToChat({ sender, text: unifiedText });

                        // Also capture any uploads attached to this payload
                        const got = extractUploadsFromRawMessage(msg);
                        if (got.length) {
                            setUploadsList(prev => {
                            const seen = new Set(prev.map(u => `${u.id}|${u.signed_url ?? ""}`));
                            const merged = [...prev];
                            for (const u of got) {
                                const item = { ...u, from: "assistant" as const, agent: sender as AgentName };
                                const key = `${item.id}|${item.signed_url ?? ""}`;
                                if (!seen.has(key)) {
                                merged.push(item);
                                seen.add(key);
                                }
                            }
                            merged.sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0));
                            return merged;
                            });
                        }
                        return;
                    }


                },
                onclose: (ev: CloseEvent) => {
                resetWebSocket();

                // Handle auth failures from the server explicitly
                if (ev.code === 4401 && !authFailedRef.current) {
                    // Try exactly one refresh + instant reconnect to avoid loops
                    authFailedRef.current = true;
                    (async () => {
                    try {
                        const res = await apiFetch('/api/refresh', { method: 'POST' });
                        if (res.ok) {
                        const { token } = await res.json();
                        try { localStorage.setItem('access_token', token); } catch {}
                        // Reset backoff and reconnect immediately
                        reconnectRef.current.tries = 0;
                        if (reconnectRef.current.timer) {
                            window.clearTimeout(reconnectRef.current.timer);
                            reconnectRef.current.timer = null;
                        }
                        setWsReconnectNonce(n => n + 1);
                        return;
                        }
                    } catch { /* ignore */ }

                    // Refresh failed → send user to sign-in
                    try { localStorage.removeItem('conversationId'); } catch {}
                    if (pathname !== '/sign-in') router.push('/sign-in');
                    })();
                    return;
                }

                // Policy violation (e.g. monthly limit) → don't auto-reconnect aggressively
                if (ev.code === 1008) {
                    // Server already delivered a user-facing message; stay disconnected
                    return;
                }
                // Hard stop on custom application closes (e.g., monthly limit)
                if (ev.code === 4000) {
                    // Show a soft indicator; do NOT reconnect (prevents flicker loops)
                    console.warn('WebSocket closed: monthly limit reached (4000). Stopping reconnects.');
                    return;
                }

                // Guard against repeated auth-fail loops (e.g., cookie blocked)
                if (ev.code === 4401 && authFailedRef.current) {
                    console.warn('Auth already retried once and still unauthorized. Stopping reconnects.');
                    return;
                }

                // Default: backoff reconnect (your existing logic)
                const base = 1000;
                const max = 15000;
                const tries = reconnectRef.current.tries;
                const nextDelay = Math.min(max, base * Math.pow(2, tries)) + Math.floor(Math.random() * 250);
                reconnectRef.current.tries = tries + 1;

                if (reconnectRef.current.timer) window.clearTimeout(reconnectRef.current.timer);
                reconnectRef.current.timer = window.setTimeout(() => setWsReconnectNonce(n => n + 1), nextDelay);
                },
                onerror: (event: Event) => {
                    console.error('WebSocket error:', event);
                    resetWebSocket();
                }
            });
            cleanup = () => {
                resetWebSocket();
                // Optional: zero-out backoff so a fresh mount starts fresh
                reconnectRef.current.tries = 0;
                if (reconnectRef.current.timer) {
                    window.clearTimeout(reconnectRef.current.timer);
                    reconnectRef.current.timer = null;
                }
            };           
        });
        return () => { if (cleanup) cleanup(); };

    }, [userToken, userName, addMessageToChat, wsReconnectNonce, conversationId, hydrateConversation, loadConversations, setTypingWithDelayAndTTL]);


    // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
    function isImageLike(mime?: string, name?: string) {
    if (mime && mime.startsWith('image/')) return true;
    if (!name) return false;
    return /\.(png|jpe?g|gif|webp|svg)$/i.test(name);
    }

    function formatBytes(n?: number | null) {
        if (!n || n <= 0) return '';
        const units = ['B','KB','MB','GB','TB'];
        const i = Math.floor(Math.log(n) / Math.log(1024));
        return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${units[i]}`;
    }



    return (
    <div className="flex h-screen w-full bg-gray-900 text-white font-sans antialiased pt-[72px]">

        {/* LEFT SIDEBAR */}
        <aside className="hidden md:flex md:flex-col fixed left-0 top-[72px] bottom-0 w-72 border-r border-gray-800 bg-gray-900 z-20">

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
        <div className="flex-1 flex flex-col md:pl-72 md:pr-72 min-w-0">
        <main
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto bg-gray-900 custom-scrollbar"
            style={{ paddingBottom: composerHeight + 24 }}
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

            {chatHistory.length > 0 && (
                <>                    

                    {chatHistory.map((msg, index) => {
                    const isUser =
                        (msg.sender || '').trim().toLowerCase() === (userName || 'You').trim().toLowerCase();
                    return (
                        <div key={index} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                        <div className={`relative p-3 md:p-4 max-w-[80%] text-white rounded-2xl md:rounded-3xl shadow-lg transition-all duration-200 ease-in-out transform hover:scale-[1.01] ${
                            isUser
                                ? 'bg-blue-600 text-white rounded-br-none md:rounded-br-none'
                                : msg.sender === 'Claude'
                                ? 'bg-gray-700 text-white rounded-bl-none md:rounded-bl-none'
                                : msg.sender === 'ChatGPT'
                                ? 'bg-gray-800 text-white rounded-bl-none md:rounded-bl-none'
                                : msg.sender === 'Gemini'
                                ? 'bg-gray-600 text-white rounded-bl-none md:rounded-bl-none'
                                : msg.sender === 'Mistral'
                                ? 'bg-green-800 text-white rounded-bl-none md:rounded-bl-none'
                                : 'bg-gray-700 text-gray-200 rounded-bl-none md:rounded-bl-none'
                        }`}>
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
                </>
            )}


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
                {/* Global fallback: only when no individual agent is showing "typing" */}
                {teamThinking && !Object.values(isTyping).some(Boolean) && (
                    <div className="mt-2 text-sm text-gray-400 italic" aria-live="polite">
                        The AI team is thinking…
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
    className="fixed bottom-0 left-0 right-0 md:left-72 md:right-72 z-50 bg-gray-900 border-t border-gray-800"
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
    <div className="flex flex-col gap-2">
        {pendingFiles.length > 0 && (
            <div className="flex flex-wrap gap-2">
                {pendingFiles.map(f => (
                    <span
                        key={f.id}
                        className="inline-flex items-center gap-2 px-2 py-1 text-xs rounded-full bg-gray-800 border border-gray-700"
                    >
                        {isImageLike(f.mime, f.name) && (
                        <img
                            src={f.signed_url ?? f.signedUrl}
                            alt={f.name}
                            className="w-6 h-6 rounded object-cover border border-gray-700"
                            loading="eager"
                            decoding="async"
                            referrerPolicy="no-referrer"
                            onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
                        />
                        )}

                        <a
                            href={f.signed_url ?? f.signedUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="truncate max-w-[180px] hover:underline"
                            title={f.name}
                        >
                            {f.name}{f.size ? ` (${formatBytes(f.size)})` : ''}
                        </a>
                        <button
                            type="button"
                            onClick={(e) => { e.preventDefault(); e.stopPropagation(); removePending(f.id); }}
                            className="text-gray-300 hover:text-white"
                            aria-label={`Remove ${f.name}`}
                            title="Remove"
                        >
                            ×
                        </button>
                    </span>
                ))}
            </div>
        )}
        <div className="flex gap-4">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFilePick}
                className="hidden"
                accept=".txt,.md,.json,.js,.ts,.tsx,.jsx,.py,.java,.go,.rb,.php,.css,.html,.sql,.yaml,.yml,.sh,.c,.cpp,.cs,.rs,.kt,image/*,.png,.jpg,.jpeg,.gif,.webp,.svg"
                multiple
            />
            <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="h-11 md:h-12 w-11 md:w-12 self-end bg-gray-700 text-white font-semibold rounded-xl hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-colors duration-200 text-lg"
                title="Attach a file"
            >
                <i className="fas fa-paperclip"></i>
            </button>
            <textarea
                ref={textareaRef}
                rows={1}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onInput={(e) => {
                    const ta = e.currentTarget;
                    ta.style.height = 'auto';
                    ta.style.height = ta.scrollHeight + 'px';
                    ta.style.overflowY = 'hidden';
                }}

                onCompositionStart={() => { isComposingRef.current = true; }}
                onCompositionEnd={() => { isComposingRef.current = false; }}
                onKeyDown={(e) => {
                    // Submit on Enter (but allow Shift+Enter for newline, and ignore while composing)
                    if (
                        e.key === 'Enter' &&
                        !e.shiftKey &&
                        !e.altKey &&
                        !e.metaKey &&
                        !e.ctrlKey &&
                        !isComposingRef.current
                    ) {
                        e.preventDefault();
                        (e.currentTarget.form as HTMLFormElement)?.requestSubmit();
                    }
                }}
                placeholder="Type your message..."
                className="flex-grow p-3 rounded-xl border border-gray-700 bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200 text-sm md:text-base resize-none overflow-hidden min-h-[44px]"
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
    </div>
</form>
</div>
{/* ===== /Composer ===== */}



        </div>
{/* RIGHT SIDEBAR — Uploaded files (always visible) */}
<aside className="hidden md:flex md:flex-col fixed right-0 top-[72px] bottom-0 w-72 border-l border-gray-800 bg-gray-900 z-20">

  <div className="p-4 border-b border-gray-800">
    <div className="font-semibold">Uploads</div>
  </div>

  <div className="flex-1 min-w-0 overflow-y-auto custom-scrollbar">
    {uploadsList.length === 0 ? (
      <div className="p-4 text-sm text-gray-400">No uploads yet</div>
    ) : (
      <ul className="w-full divide-y divide-gray-800 flex flex-col">

        {uploadsList.map((u, i) => (
          <li key={`${u.id}-${i}`} className="px-3 py-2">
            <div className="flex items-center justify-between gap-3">
                {isImageLike(u.mime, u.name) && u.signed_url && (
                <img
                    src={u.signed_url}
                    alt={u.name}
                    className="w-10 h-10 rounded object-cover border border-gray-700 flex-shrink-0"
                    loading="lazy"
                    decoding="async"
                    referrerPolicy="no-referrer"
                    onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }}
                />
                )}

                <a
                    href="#"
                    className="truncate hover:underline"
                    title={u.name}
                    onClick={async (e) => {
                        e.preventDefault();
                        try {
                        // Always ask the backend for a fresh signed URL
                        const res = await apiFetch(`/api/uploads/${encodeURIComponent(u.id)}/url`, { cache: 'no-store' });
                        if (!res.ok) throw new Error(`HTTP ${res.status}`);
                        const data = await res.json();
                        const fresh = (data && (data.signed_url || data.url)) as string | undefined;
                        if (fresh) {
                            window.open(fresh, '_blank', 'noopener,noreferrer');
                        } else {
                            alert('Could not get a download link. Try reloading the conversation or re-uploading the file.');
                        }
                        } catch (err) {
                            console.error('open upload failed', err);
                            alert('Link expired and could not be refreshed. Please reload the conversation or re-upload.');
                        }
                    }}
                >
                    {u.name}
                </a>

            </div>
            <div className="mt-1 text-[11px] text-gray-400">
              {(u.created_at ? new Date(u.created_at).toLocaleString() : '')}
              {typeof u.size === 'number' ? ` • ${formatBytes(u.size)}` : ''}
              {u.agent ? ` • by ${u.agent}` : (u.from ? ` • ${u.from}` : '')}
            </div>
          </li>
        ))}
      </ul>
    )}
  </div>
</aside>        
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