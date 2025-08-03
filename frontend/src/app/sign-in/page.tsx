// src/app/sign-in/page.tsx
'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '../../context/UserContext';
import { GoogleLogin, CredentialResponse } from '@react-oauth/google';

// This component handles the Google Sign-In process.
export default function SignInPage() {
    const { userToken, setUserToken, setUserName } = useUser();
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(false);
    // The clientId is now read from the environment variable and used by GoogleOAuthProvider in layout.tsx
    const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || '';

    // Effect to check if the user is already logged in
    useEffect(() => {
        if (userToken) {
            router.push('/chat'); // Redirect to chat page if already authenticated
        }
    }, [userToken, router]);

    // Handle successful sign-in with Google
    const handleGoogleSuccess = async (response: CredentialResponse) => {
        setIsLoading(true);
        const { credential } = response;
        if (!credential) {
            console.error("Google login failed: No credential received.");
            setIsLoading(false);
            return;
        }

        try {
            const backendUrl = process.env.NEXT_PUBLIC_WS_URL;
            if (!backendUrl) {
                console.error("NEXT_PUBLIC_WS_URL environment variable is not set.");
                return;
            }
            
            const res = await fetch(`${backendUrl}/api/google-auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ id_token: credential }),
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

        } catch (error) {
            console.error('Sign-in failed:', error);
            setIsLoading(false);
        }
    };

    // Handle failed sign-in with Google
    const handleGoogleFailure = (error: Error) => {
        console.error('Google sign-in failed:', error);
        setIsLoading(false);
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-950 text-white">
            <div className="text-center p-8 bg-gray-900 rounded-2xl shadow-xl max-w-md w-full">
                <h1 className="text-4xl font-extrabold mb-4 text-white">Welcome to Colosseum</h1>
                <p className="text-lg text-gray-400 mb-8">
                    Sign in to start a conversation with your AI team.
                </p>
                {isLoading ? (
                    <div className="flex justify-center items-center h-12">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                ) : (
                    <GoogleLogin
                        onSuccess={handleGoogleSuccess}
                        onError={() => handleGoogleFailure(new Error("Google sign-in failed."))}
                    />
                )}
            </div>
        </div>
    );
}