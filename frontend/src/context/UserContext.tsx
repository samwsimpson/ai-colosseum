'use client';

import React, { createContext, useState, useEffect, useContext } from 'react';
import { useRouter } from 'next/navigation';

interface UserContextType {
  userName: string | null;
  userToken: string | null;
  setUserName: (name: string | null) => void;
  setUserToken: (token: string | null) => void;
  handleLogout: () => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [userName, setUserName] = useState<string | null>(null);
  const [userToken, setUserToken] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const storedToken = localStorage.getItem('access_token');
    if (storedToken) {
      const fetchUserData = async () => {
        try {
          const response = await fetch('http://localhost:8000/api/users/me', {
            headers: {
              'Authorization': `Bearer ${storedToken}`,
            },
          });
          if (!response.ok) {
            throw new Error('Token validation failed.');
          }
          const data = await response.json();
          setUserName(data.user_name);
          setUserToken(storedToken);
        } catch (error) {
          console.error('Failed to validate token:', error);
          handleLogout();
        }
      };
      fetchUserData();
    }
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_name');
    localStorage.removeItem('user_id');
    setUserName(null);
    setUserToken(null);
    router.push('/');
  };

  const value = { userName, userToken, setUserName, setUserToken, handleLogout };

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