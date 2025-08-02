'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '../../context/UserContext';
import { GoogleOAuthProvider, GoogleLogin, CredentialResponse } from '@react-oauth/google';

// This component handles the Google Sign-In process.
export default function SignInPage() {
    const { userToken, setUserToken, setUserName } = useUser();
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(false);
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
        // The Google response object contains a `credential` (id_token) in this flow
        const { credential } = response;
        if (!credential) {
            console.error("Google login failed: No credential received.");
            setIsLoading(false);
            return;
        }

        try {
            // Send the id_token to your backend for verification
            const res = await fetch('http://localhost:8000/api/google-auth', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // The body now sends a simple object with the id_token
                body: JSON.stringify({ id_token: credential }),
            });

            if (!res.ok) {
                const errorData = await res.json();
                console.error("Backend error response:", errorData);
                throw new Error('Google authentication failed on backend.');
            }

            const data = await res.json();
            // Store the access token and user info directly in state and local storage
            localStorage.setItem('userToken', data.access_token);
            localStorage.setItem('userName', data.user_name);
            setUserToken(data.access_token);
            setUserName(data.user_name);

            // Introduce a small delay to ensure state is fully updated before redirecting
            setTimeout(() => {
                 router.push('/chat'); 
            }, 500); // 500ms delay
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
        <GoogleOAuthProvider clientId={googleClientId}>
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
                        // This uses the simpler `CredentialResponse` flow
                        <GoogleLogin
                            onSuccess={handleGoogleSuccess}
                            onError={handleGoogleFailure}
                        />
                    )}
                </div>
            </div>
        </GoogleOAuthProvider>
    );
}

