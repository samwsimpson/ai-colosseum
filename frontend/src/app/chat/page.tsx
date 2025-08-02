'use client';

import { useState, FormEvent, useRef, useEffect, useCallback, useMemo } from 'react';
import React from 'react';
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

export default function ChatPage() {
    const { userName, userToken, clearUser } = useUser();
    const router = useRouter();
    const pathname = usePathname();

    const [message, setMessage] = useState<string>('');
    const [chatHistory, setChatHistory] = useState<Message[]>([]);
    const [isWsOpen, setIsWsOpen] = useState<boolean>(false);
    // New state to track which agents are typing
    const [isTyping, setIsTyping] = useState<TypingState>({
        ChatGPT: false,
        Claude: false,
        Gemini: false,
        Mistral: false,
    });
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const chatEndRef = useRef<HTMLDivElement>(null);
    const ws = useRef<WebSocket | null>(null);

    // A simple ref to distinguish the first render
    const isInitialRender = useRef(true);
    
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
            const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
            const socket = new WebSocket(`${wsUrl}/ws/colosseum-chat?token=${userToken}`);
            ws.current = socket;
            setIsWsOpen(false); // Set to false while connecting
        }
        
        // Create a stable local reference to the current WebSocket instance
        const currentWs = ws.current; 

        currentWs.onopen = () => {
            console.log('WebSocket connection opened.');
            setIsWsOpen(true);
            const initialPayload = {
                message: "Hello!",
                user_name: userName 
            };
            currentWs.send(JSON.stringify(initialPayload));
        };

        currentWs.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                if (typeof message.typing === 'boolean') {
                    console.log(`${message.sender} is typing: ${message.typing}`);
                    setIsTyping(prev => ({
                        ...prev,
                        [message.sender]: message.typing,
                    }));
                } 
                else if (message.text) {
                    addMessageToChat(message);
                }
            } catch (error) {
                console.error("Failed to parse message:", error);
                addMessageToChat({ sender: "System", text: `Error: ${event.data}` });
            }
        };

        currentWs.onclose = () => {
            console.log('WebSocket connection closed.');
            setIsWsOpen(false);
            setIsTyping({ ChatGPT: false, Claude: false, Gemini: false, Mistral: false });
        };
        
        currentWs.onerror = (error) => {
            console.error('WebSocket error:', error);
            setIsWsOpen(false);
        };

        // Cleanup function for the useEffect hook
        return () => {
            // Close the socket only if it's the one we just opened and it's still open
            if (currentWs && currentWs.readyState === WebSocket.OPEN) {
                console.log('Component unmounting, closing WebSocket.');
                currentWs.close();
            }
        };
    }, [userToken, userName, addMessageToChat]);

    if (!userToken) {
        return null;
    }
    
    return (
        <div className="flex flex-col h-screen w-full bg-gray-950 text-white font-sans antialiased">
            <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 bg-gray-900 custom-scrollbar">
                <div className="max-w-4xl mx-auto flex flex-col space-y-6">
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
                                className={`flex ${msg.sender === (userName || 'You') ? 'justify-end' : 'justify-start'} ${index === 0 ? 'mt-44' : ''}`}
                            >
                                <div className={`relative p-4 max-w-2xl text-white rounded-3xl shadow-lg transition-all duration-200 ease-in-out transform hover:scale-[1.01] ${
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
                            <div className="flex items-center space-x-2 px-4 py-3 rounded-2xl max-w-xs bg-gray-800 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">ChatGPT is typing...</span>
                            </div>
                        )}
                        {isTyping.Claude && (
                            <div className="flex items-center space-x-2 px-4 py-3 rounded-2xl max-w-xs bg-gray-700 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">Claude is typing...</span>
                            </div>
                        )}
                        {isTyping.Gemini && (
                            <div className="flex items-center space-x-2 px-4 py-3 rounded-2xl max-w-xs bg-purple-800 text-white shadow-md">
                                <div className="flex space-x-1">
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-fast"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slow"></span>
                                    <span className="h-2 w-2 bg-gray-400 rounded-full animate-pulse-slower"></span>
                                </div>
                                <span className="text-sm text-gray-300 font-medium">Gemini is typing...</span>
                            </div>
                        )}
                        {isTyping.Mistral && (
                            <div className="flex items-center space-x-2 px-4 py-3 rounded-2xl max-w-xs bg-green-800 text-white shadow-md">
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
                className="fixed bottom-0 left-0 right-0 z-10 bg-gray-900 border-t border-gray-800 p-6 shadow-lg"
            >
              
                <div className="flex gap-4 max-w-4xl mx-auto">
                
                    
                        
                    <input
                        type="text"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-grow p-3 rounded-xl border border-gray-700 bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-200"
                        disabled={!isWsOpen || !userName}
                    />
                    <button
                        type="submit"
                        className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 transition-colors duration-200"
                        disabled={!isWsOpen || !userName}
                    >
                        Send
                    </button>
                </div>
            </form>

            <style jsx>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: #2d3748;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: #4a5568;
                    border-radius: 20px;
                    border: 3px solid #2d3748;
                }
                @keyframes pulse-fast {
                    0%, 100% { opacity: 0.5; }
                    50% { opacity: 1; }
                }
                @keyframes pulse-slow {
                    0%, 100% { opacity: 0.5; }
                    50% { opacity: 1; }
                }
                @keyframes pulse-slower {
                    0%, 100% { opacity: 0.5; }
                    50% { opacity: 1; }
                }
                .animate-pulse-fast { animation: pulse-fast 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
                .animate-pulse-slow { animation: pulse-slow 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite 0.2s; }
                .animate-pulse-slower { animation: pulse-slower 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite 0.4s; }
            `}</style>
        </div>
    );
}