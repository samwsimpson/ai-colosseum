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
  type?: 'ping' | 'pong' | string;
}

type HBWebSocket = WebSocket & {
  _heartbeatInterval?: number | null;
  _lastPongAt?: number;
  _pingWatchdog?: number | null;
};
export default function ChatPage() {
    const { userName, userToken } = useUser();
    const router = useRouter();
    const pathname = usePathname();

    const [message, setMessage] = useState<string>('');
    const [chatHistory, setChatHistory] = useState<Message[]>([]);
    const [isWsOpen, setIsWsOpen] = useState<boolean>(false);
    const [wsReconnectNonce, setWsReconnectNonce] = useState(0);
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
    
    // Handle redirection to sign-in page when userToken is not present
    useEffect(() => {
      if (!userToken && pathname !== '/sign-in') {
        router.push('/sign-in');
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

    // WebSocket connection logic
    useEffect(() => {
        // Guard clause to prevent connection without a token
        if (!userToken) {
            if (ws.current) {
                ws.current.close();
            }
            ws.current = null;
            setIsWsOpen(false);
            return;
        }

        // Only create a new WebSocket if one does not exist or is already closed
        if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
            const rawBase = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
            // remove scheme + trailing slash cleanly
            const normalizedBase = rawBase.replace(/^https?:\/\//, '').replace(/\/+$/, '');
            const wsProtocol = rawBase.startsWith('https://') ? 'wss://' : 'ws://';
            const wsUrl = `${wsProtocol}${normalizedBase}/ws/colosseum-chat?token=${userToken}`;

            console.log('[WS] dialing:', wsUrl); // <— SEE EXACT URL
            const socket = new WebSocket(wsUrl);
            ws.current = socket;
            setIsWsOpen(false);
        }
        
        // Create a stable local reference to the current WebSocket instance
        const currentWs = ws.current; 

        currentWs.onopen = () => {
            console.log('WebSocket connection opened.');
            setIsWsOpen(true);

            // --- HEARTBEAT to keep connection alive ---
            const heartbeatInterval = window.setInterval(() => {
                if ((currentWs as HBWebSocket)._lastPongAt && Date.now() - (currentWs as HBWebSocket)._lastPongAt! > 60000) {
                    currentWs.close();
                    return;
                }
                currentWs.send(JSON.stringify({ sender: "System", type: "ping" }));
            }, 20000);
            // store so we can clear on close
            (currentWs as HBWebSocket)._heartbeatInterval = heartbeatInterval;
            // --- END HEARTBEAT ---

            const initialPayload = {
                message: "Hello!",
                user_name: userName 
            };
            currentWs.send(JSON.stringify(initialPayload));
        };

        currentWs.onmessage = (event: MessageEvent<string>) => {
            try {
                const parsed: unknown = JSON.parse(event.data);

                const msg = parsed as ServerMessage;
                
                if (msg.type === 'pong') {
                    (currentWs as HBWebSocket)._lastPongAt = Date.now();
                    return; // swallow
                }

                // reply to server pings (skip showing them)
                if (msg.type === 'ping') {
                    if (currentWs.readyState === WebSocket.OPEN) {
                        currentWs.send(JSON.stringify({ type: 'pong' }));
                    }
                    return;
                }

                if (typeof msg.sender === 'string' && typeof msg.typing === 'boolean') {
                    const senderKey: string = msg.sender;
                    const isTypingVal: boolean = msg.typing;

                    setIsTyping((prev: TypingState): TypingState => {
                        const next: TypingState = { ...prev };
                        next[senderKey] = isTypingVal;
                        return next;
                    });
                    return;
                }

                if (typeof msg.sender === 'string' && typeof msg.text === 'string') {
                    addMessageToChat({ sender: msg.sender, text: msg.text });
                }

            } catch (err: unknown) {
                console.error('Failed to parse message:', err);
                addMessageToChat({ sender: 'System', text: `Error: ${event.data}` });
            }
        };
                
        currentWs.onclose = (evt: CloseEvent) => {
            console.log('WebSocket closed.', 'code=', evt.code, 'reason=', evt.reason, 'wasClean=', evt.wasClean);
            setIsWsOpen(false);
            setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });

            // clear heartbeat if we set one
            const hb = (currentWs as HBWebSocket)._heartbeatInterval;
            if (typeof hb === 'number') window.clearInterval(hb);

            // mark ref as closed and trigger reconnect
            ws.current = null;
            setTimeout(() => setWsReconnectNonce(n => n + 1), 1000); // retry in 1s
        };
        
        currentWs.onerror = (event: Event) => {
            console.error('WebSocket error:', event);
            setIsWsOpen(false);
        };

        // Cleanup function for the useEffect hook
        return () => {
            const hb = (currentWs as HBWebSocket)._heartbeatInterval;
            if (hb) window.clearInterval(hb);

            if (currentWs && (currentWs.readyState === WebSocket.OPEN || currentWs.readyState === WebSocket.CONNECTING)) {
                currentWs.close();
            }
        };
    }, [userToken, userName, addMessageToChat, wsReconnectNonce]);

    if (!userToken) {
        return null;
    }
    
    return (
        <div className="flex flex-col h-screen w-full bg-gray-950 text-white font-sans antialiased">
            <main ref={chatContainerRef} className="flex-1 overflow-y-auto pt-[72px] pb-[100px] bg-gray-900 custom-scrollbar">
                <div className="max-w-4xl mx-auto flex flex-col space-y-4 md:space-y-6">
                    {/* Conditional rendering for the initial message */}
                    {!isWsOpen && userToken ? (
                        <p className="text-center text-gray-500 py-12 text-lg">Connecting to chat...</p>
                    ) : (
                        chatHistory.length === 0 && (
                            <p className="text-center text-gray-500 py-12 text-lg">
                                {userName ? `Start a conversation with the AI team, ${userName}...` : `Please sign in to start a conversation...`}
                            </p>
                        )
                    )}

                    {chatHistory.length > 0 && (
                        chatHistory.map((msg, index) => (
                            <div
                                key={index}
                                className={`flex ${msg.sender === (userName || 'You') ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`relative p-3 md:p-4 max-w-[80%] text-white rounded-2xl md:rounded-3xl shadow-lg transition-all duration-200 ease-in-out transform hover:scale-[1.01] ${
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
                                }`}>
                                    {msg.sender !== (userName || 'You') && (
                                        <div className="text-xs text-gray-300 mb-1 font-semibold">
                                            {msg.sender}
                                        </div>
                                    )}
                                    <pre className="whitespace-pre-wrap font-sans text-sm md:text-base leading-relaxed">{msg.text}</pre>
                                </div>
                            </div>
                        ))
                    )}
                    <div className="flex flex-col space-y-2">
                        {isTyping.ChatGPT && (
                            <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-gray-800 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">ChatGPT is typing...</span>
                            </div>
                        )}
                        {isTyping.Claude && (
                            <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-gray-700 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">Claude is typing...</span>
                            </div>
                        )}
                        {isTyping.Gemini && (
                            <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-purple-800 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">Gemini is typing...</span>
                            </div>
                        )}
                        {isTyping.Mistral && (
                            <div className="flex items-center space-x-2 px-3 py-2 md:px-4 md:py-3 rounded-2xl max-w-xs bg-green-800 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">Mistral is typing...</span>
                            </div>
                        )}
                    </div>
                    <div ref={chatEndRef} />
                </div>
            </main>

            <form
                onSubmit={handleSubmit}
                className="fixed bottom-0 left-0 right-0 z-10 bg-gray-900 border-t border-gray-800 p-4 md:p-6 shadow-lg"
            >
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
                </div>
            </form>
        </div>
    );
}