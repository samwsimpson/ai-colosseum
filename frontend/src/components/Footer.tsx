// frontend/src/components/Footer.tsx
import Link from 'next/link';

const Footer = () => {
  return (
    // The 'mt-auto' class pushes the footer to the bottom of the page in a flex container
    // The 'items-center' class vertically centers the content of the grid columns
    <footer className="w-full bg-gray-900 text-gray-400 py-12 md:py-16 mt-auto">
      <div className="container mx-auto px-4 md:px-8 grid grid-cols-1 md:grid-cols-3 gap-8 items-center text-center md:text-left">
        {/* Left Column: Logo */}
        <div className="flex justify-center md:justify-start">
          <Link href="/">
            <img src="/ColosseumLogoBox.svg" alt="The Colosseum Logo" className="h-[200px] w-auto" />
          </Link>
        </div>

        {/* Center Column: Copyright and Links */}
        <div className="flex flex-col items-center space-y-2 text-center">
          <p className="text-sm">
            &copy; {new Date().getFullYear()} The AI Colosseum. All rights reserved.
          </p>
          <p className="text-sm">
            A{' '}
            <a href="https://kumokodo.ai/" className="text-blue-400 hover:underline">
              KumoKodo.ai
            </a>{' '}
            SaaS Application
          </p>
          <div className="flex space-x-4 text-sm mt-2">
            <Link href="/privacy-policy" className="hover:underline">
              Privacy Policy
            </Link>
            <Link href="/terms-of-service" className="hover:underline">
              Terms of Service
            </Link>
          </div>
        </div>

        {/* Right Column: Vertical Nav */}
        <div className="flex flex-col items-center md:items-end space-y-2 text-center md:text-right">
          <h4 className="text-lg font-bold text-white mb-2">Navigation</h4>
          <Link href="/" className="hover:text-white transition-colors duration-200">
            Home
          </Link>
          <Link href="/chat" className="hover:text-white transition-colors duration-200">
            Chat
          </Link>
          <Link href="/contact" className="hover:text-white transition-colors duration-200">
            Contact
          </Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
