// frontend/src/components/LayoutWrapper.tsx
'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { Header } from './Header';
import Footer from './Footer';

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <>
      <Header />
      <main className="flex-1 pt-20">
        {children}
      </main>
      {pathname !== '/chat' && <Footer />}
    </>
  );
}
