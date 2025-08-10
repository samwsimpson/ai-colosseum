'use client';

import { useState, FormEvent, useRef, useEffect, useCallback } from 'react';
import React from 'react';
import { useUser } from '../../context/UserContext';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useGoogleLogin } from '@react-oauth/google';

// This component handles the Google Sign-In process.
export default function SignInPage() {
    const { userToken, handleLogin } = useUser();
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const [isLoading, setIsLoading] = useState(false);

    // This useEffect handles the OAuth callback from Google
    useEffect(() => {
        const code = searchParams.get('code');
        const redirect_uri = window.location.origin + pathname;
        
        if (code && !userToken) {
            handleGoogleSuccess({ code, redirect_uri });
        }
    }, [searchParams, userToken, pathname, handleGoogleSuccess]);

    // This function sends the authorization code to the backend
    const handleGoogleSuccess = useCallback(async (tokenResponse) => {
        setIsLoading(true);
        try {
            const backendUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
            const response = await fetch(`${backendUrl}/api/google-auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: tokenResponse.code,
                    redirect_uri: tokenResponse.redirect_uri
                }),
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`Google authentication failed on backend. Details: ${JSON.stringify(errorData)}`);
            }
            
            const tokenData = await response.json();
            handleLogin(tokenData);
            router.push('/chat');

        } catch (error) {
            console.error('Sign-in failed:', error);
            setIsLoading(false);
        }
    }, [handleLogin, router]);

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
    
    // Redirect if user is already logged in
    useEffect(() => {
        if (userToken) {
            router.push('/chat');
        }
    }, [userToken, router]);
    
    // Don't render the sign-in button if we are redirecting or already logged in
    if (userToken || isLoading) {
      return null;
    }

    return (
        <div className="flex flex-col min-h-screen bg-gray-950 text-white font-sans antialiased items-center justify-center p-6">
            <main className="flex flex-col items-center justify-center space-y-8 text-center p-8 bg-gray-900 rounded-3xl shadow-2xl max-w-lg w-full">
                <h1 className="text-4xl font-extrabold text-white">Sign In</h1>
                <p className="text-lg text-gray-400">
                    Join The Colosseum to orchestrate a team of powerful AIs.
                </p>
                <button
                    onClick={() => login()}
                    className="flex items-center justify-center w-full px-6 py-3 border border-transparent text-base font-semibold rounded-xl text-gray-900 bg-white hover:bg-gray-200 transition-colors duration-200 shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                    <svg className="w-5 h-5 mr-2" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M43.611 20.082H42V20H24v8h11.303c-1.619 6.757-7.237 10.158-11.303 10.158-6.663 0-12.194-5.49-12.194-12.247s5.531-12.247 12.194-12.247c3.155 0 5.865 1.498 7.93 3.655l5.518-5.253C34.453 6.945 29.742 4 24 4 12.51 4 3.013 13.513 3.013 24s9.497 20 20.987 20c11.006 0 20.198-8.293 20.198-19.882 0-1.393-.204-2.812-.497-4.136z" fill="#FFC107" />
                        <path d="M43.611 20.082L42 20H24v8h11.303a11.968 11.968 0 01-4.908 6.938c-2.45 1.944-5.787 3.062-9.423 3.062-6.663 0-12.194-5.49-12.194-12.247s5.531-12.247 12.194-12.247c3.155 0 5.865 1.498 7.93 3.655l5.518-5.253C34.453 6.945 29.742 4 24 4 12.51 4 3.013 13.513 3.013 24s9.497 20 20.987 20c11.006 0 20.198-8.293 20.198-19.882 0-1.393-.204-2.812-.497-4.136z" fill="#4285F4" />
                        <path d="M12.194 36.153S10.255 36.143 8.36 35.808c-1.895-.335-3.69-1.296-5.111-2.717l4.474-3.468C7.755 30.56 9.615 31.42 12.194 31.42c2.579 0 4.439-1.27 6.302-3.415l4.316 3.993-5.518 5.253z" fill="#FBBC05" />
                        <path d="M24 20.082a11.968 11.968 0 01-4.908 6.938c-2.45 1.944-5.787 3.062-9.423 3.062s-6.973-1.118-9.423-3.062a11.968 11.968 0 01-4.908-6.938L3.013 24c0 10.487 9.497 20 20.987 20s20.987-9.513 20.987-20H24z" fill="#EA4335" />
                    </svg>
                    Sign in with Google
                </button>
            </main>
        </div>
    );
}

