// src/app/sign-in/page.tsx
'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '../../context/UserContext';
import { useGoogleLogin, CodeResponse } from '@react-oauth/google';

// This component handles the Google Sign-In process.
export default function SignInPage() {
    const { userToken, setUserToken, setUserName } = useUser();
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(false);

    // Effect to check if the user is already logged in
    useEffect(() => {
        if (userToken) {
            router.push('/chat'); // Redirect to chat page if already authenticated
        }

        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');

        if (code) {
            // We have a code, so we can exchange it for a token.
            // The `handleGoogleSuccess` function is already set up to do this.
            // We need to pass it an object with the code.
            handleGoogleSuccess({ code });
        }
    }, [userToken, router]);

    // Handle successful sign-in with Google
    const handleGoogleSuccess = async ({ code }: { code: string }) => {
        setIsLoading(true);

        try {
            const backendUrl = process.env.NEXT_PUBLIC_WS_URL;
            if (!backendUrl) {
                console.error("NEXT_PUBLIC_WS_URL environment variable is not set.");
                setIsLoading(false);
                return;
            }
            
            const res = await fetch(`${backendUrl}/api/google-auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code }),
            });

            if (!res.ok) {
                const errorData = await res.json();
                console.error("Backend error response:", errorData);
                throw new Error('Google authentication failed on backend.');
            }

            const data = await res.json();
            localStorage.setItem('userToken', data.access_token);
            localStorage.setItem('userName', data.user_name);
            setUserToken(data.access_token);
            setUserName(data.user_name);
            // On successful auth, redirect to chat
            router.push('/chat');

        } catch (error) {
            console.error('Sign-in failed:', error);
            setIsLoading(false);
        }
    };

    // Handle failed sign-in with Google
    const handleGoogleFailure = () => {
        console.error('Google sign-in failed.');
        setIsLoading(false);
    };

    const login = useGoogleLogin({
        onSuccess: handleGoogleSuccess,
        onError: handleGoogleFailure,
        flow: 'auth-code',
        ux_mode: 'redirect',
        redirect_uri: typeof window !== 'undefined' ? window.location.origin + '/sign-in' : '',
    });

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-950 text-white">
            <div className="text-center p-8 bg-gray-900 rounded-2xl shadow-xl max-w-md w-full">
                <h1 className="text-4xl font-extrabold mb-4 text-white">Welcome to The AI Colosseum</h1>
                <p className="text-lg text-gray-400 mb-8">
                    Sign in to start a conversation with your AI team.
                </p>
                {isLoading ? (
                    <div className="flex justify-center items-center h-12">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                ) : (
                    <button
                        onClick={() => login()}
                        className="flex items-center justify-center bg-white text-gray-700 font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-gray-100 transition-colors"
                    >
                        <svg className="w-6 h-6 mr-2" viewBox="0 0 48 48">
                            <path fill="#4285F4" d="M24 9.5c3.97 0 7.21 1.44 9.86 3.87l-4.4 4.4c-.98-.94-2.5-2.05-5.46-2.05-4.17 0-7.85 2.87-9.13 6.73H1.45v-4.6C4.32 12.94 10.45 9.5 24 9.5z"/>
                            <path fill="#34A853" d="M46.5 24.5c0-1.56-.14-3.06-.4-4.5H24v8.5h12.8c-.55 2.7-2.17 5.03-4.64 6.6v5.5h7.1c4.16-3.83 6.54-9.57 6.54-16.1z"/>
                            <path fill="#FBBC05" d="M14.87 28.18c-.48-1.44-.77-3-.77-4.68s.29-3.24.77-4.68v-5.5H7.77C6.17 16.3 5.5 19.95 5.5 24s.67 7.7 2.27 11.18l7.1-5.5z"/>
                            <path fill="#EA4335" d="M24 48c6.48 0 12.02-2.13 16.03-5.79l-7.1-5.5c-2.13 1.44-4.87 2.29-8.93 2.29-5.83 0-10.8-3.5-12.55-8.26H1.45v5.5C4.32 42.62 10.45 48 24 48z"/>
                            <path fill="none" d="M0 0h48v48H0z"/>
                        </svg>
                        Sign in with Google
                    </button>
                )}
            </div>
        </div>
    );
}