'use client';

import Link from 'next/link';
import { useUser } from '../context/UserContext';
import { useRouter } from 'next/navigation';

export const Header = () => {
  const { userName, handleLogout } = useUser();
  const router = useRouter();

  return (
    // The 'fixed' class makes the header sticky at the top of the viewport.
    // 'top-0 left-0 right-0' pins it to all sides.
    // 'z-50' ensures it stays on top of other content.
    <header className="fixed top-0 left-0 right-0 z-50 p-4 bg-gray-800 text-white shadow-lg w-full">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-4">
          {/* NEW: Replaced h1 with the SVG logo and made it taller */}
          <Link href="/">
            <img src="/ColosseumLogo.svg" alt="The Colosseum" className="h-10 w-auto" />
          </Link>
        </div>
        <nav className="space-x-4 flex items-center">
          <Link href="/" className="hover:text-blue-400">Home</Link>
          <Link href="/chat" className="hover:text-blue-400">Chat</Link>
          <Link href="/contact" className="hover:text-blue-400">Contact</Link>
          {userName ? (
            <>
              <span className="font-bold">Welcome, {userName}!</span>
              <button
                onClick={() => {
                  handleLogout();
                  router.push('/sign-in');
                }}
                className="bg-red-600 text-white font-bold py-2 px-4 rounded-full hover:bg-red-700 transition duration-300"
              >
                Logout
              </button>
            </>
          ) : (
            <button
                onClick={() => router.push('/sign-in')}
                className="bg-blue-600 text-white font-bold py-2 px-4 rounded-full hover:bg-blue-700 transition duration-300"
              >
              Sign In
            </button>
          )}
        </nav>
      </div>
    </header>
  );
};

export default Header;
