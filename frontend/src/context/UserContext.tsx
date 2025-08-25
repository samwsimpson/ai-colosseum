// src/context/UserContext.tsx
'use client';

import React, { createContext, useState, useEffect, useContext } from 'react';
import { useRouter } from 'next/navigation';

interface UserContextType {
  userName: string | null;
  userToken: string | null;
  userId: string | null;
  setUserName: (name: string | null) => void;
  setUserToken: (token: string | null) => void;
  setUserId: (id: string | null) => void;
  handleLogout: () => void;
  handleLogin: (payload: { access_token: string; user_name: string; user_id: string }) => void;
}
const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
    const [userName, setUserName] = useState<string | null>(null);
    const [userToken, setUserToken] = useState<string | null>(null);
    const router = useRouter();
    const [userId, setUserId] = useState<string | null>(null);
    useEffect(() => {
        const storedToken = localStorage.getItem('userToken');
        if (storedToken && !userToken) {
        // Token exists in local storage but not in state.
        // Set the token to trigger the chat page's logic.
        setUserToken(storedToken);
        }

        const storedName = localStorage.getItem('userName');
        if (storedName && !userName) {
        setUserName(storedName);
        }

        const storedId = localStorage.getItem('userId');
        if (storedId && !userId) {
        setUserId(storedId);
        }

        // Now, validate the token only if it is present.
        if (userToken) {
            const validateToken = async () => {
                const backendUrl = process.env.NEXT_PUBLIC_WS_URL;
                if (!backendUrl) {
                    console.error('NEXT_PUBLIC_WS_URL environment variable is not set.');
                    return;
                }
                try {
                    const response = await fetch(`${backendUrl}/api/users/me`, {
                        headers: { Authorization: `Bearer ${userToken}` },
                    });
                    if (!response.ok) {
                        throw new Error('Token validation failed.');
                    }
                } catch (error) {
                    console.error('Error fetching user data:', error);
                    handleLogout(); // Automatically log out on token failure
                }
            };
            validateToken();
        }
    }, [userToken, userName, userId, handleLogout]);

  // ➕ added implementation
  const handleLogin = (payload: { access_token: string; user_name: string; user_id: string }) => {
    try {
      localStorage.setItem('userToken', payload.access_token);
      localStorage.setItem('userName', payload.user_name);
      localStorage.setItem('userId', payload.user_id);
    } catch {
      /* ignore storage errors */
    }

    setUserToken(payload.access_token);
    setUserName(payload.user_name);
    setUserId(payload.user_id);
  };

const handleLogout = () => {
    localStorage.removeItem('userToken');
    localStorage.removeItem('userName');
    localStorage.removeItem('userId'); // ➕ NEW: Clear userId from local storage
    setUserName(null);
    setUserToken(null);
    setUserId(null); // ➕ NEW: Clear userId from state
    router.push('/');
  };
  
  const value = {
    userName,
    userToken,
    setUserName,
    setUserToken,
    setUserId,
    handleLogout,
    handleLogin, // ➕ expose it
  };

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

export default UserContext;
