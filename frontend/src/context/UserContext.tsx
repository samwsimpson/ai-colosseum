// src/context/UserContext.tsx
'use client';

import React, { createContext, useState, useEffect, useContext } from 'react';
import { useRouter } from 'next/navigation';

interface UserContextType {
  userName: string | null;
  userToken: string | null;
  setUserName: (name: string | null) => void;
  setUserToken: (token: string | null) => void;
  handleLogout: () => void;
  handleLogin: (payload: { access_token: string; user_name: string; user_id: string }) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [userName, setUserName] = useState<string | null>(null);
  const [userToken, setUserToken] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const backendUrl = process.env.NEXT_PUBLIC_WS_URL;
    if (!backendUrl) {
      console.error('NEXT_PUBLIC_WS_URL environment variable is not set.');
      return;
    }

    const storedToken = localStorage.getItem('userToken');
    if (storedToken) {
      const fetchUserData = async () => {
        try {
          const response = await fetch(`${backendUrl}/api/users/me`, {
            headers: { Authorization: `Bearer ${storedToken}` },
          });
          if (!response.ok) throw new Error('Token validation failed.');
          const data = await response.json();
          setUserName(data.user_name ?? null);
          setUserToken(storedToken);
        } catch (error) {
          console.error('Error fetching user data:', error);
          localStorage.removeItem('userToken');
          localStorage.removeItem('userName');
          setUserName(null);
          setUserToken(null);
        }
      };
      fetchUserData();
    }
  }, [router]);

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
  };

  const handleLogout = () => {
    localStorage.removeItem('userToken');
    localStorage.removeItem('userName');
    setUserName(null);
    setUserToken(null);
    router.push('/');
  };

  const value = {
    userName,
    userToken,
    setUserName,
    setUserToken,
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
