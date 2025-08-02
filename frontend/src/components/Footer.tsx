// frontend/src/components/Footer.tsx
const Footer = () => {
  return (
    <footer className="w-full bg-gray-900 text-gray-500 py-6 mt-auto">
      <div className="container mx-auto text-center">
        <p>&copy; {new Date().getFullYear()} The Colosseum. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;