import { useState, useEffect } from 'react';
import { Car, Github, Terminal, Sun, Moon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [activeSection, setActiveSection] = useState('');

  // Scroll effect (unchanged)
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Load saved theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      document.documentElement.classList.add('dark');
      setIsDark(true);
    }
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['features', 'about', 'opensource'];
    
      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          if (rect.top <= 120 && rect.bottom >= 120) {
            setActiveSection(section);
          }
        }
      }
    };
  
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Toggle dark / light mode
  const toggleTheme = () => {
    const nextIsDark = !isDark;
    setIsDark(nextIsDark);

    if (nextIsDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  return (
    <div className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4 pointer-events-none">
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="
          pointer-events-auto flex items-center gap-2 p-2 rounded-2xl
          bg-white/80 dark:bg-[#0a0a0a]/80
          backdrop-blur-2xl
          border border-black/10 dark:border-white/10
          shadow-2xl shadow-black/20
        "
      >
        {/* Brand */}
        <Link
          to="/"
          className="flex items-center gap-2 px-4 py-2 rounded-xl group hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        >
          <div className="text-blue-500 group-hover:text-blue-400 transition-colors">
            <Car size={20} strokeWidth={2.5} />
          </div>
          <span className="font-bold tracking-tight text-gray-800 dark:text-gray-200">
            DriveDetect
          </span>
        </Link>

        <div className="w-px h-6 bg-black/10 dark:bg-white/10 mx-2 hidden sm:block" />

        {/* Links */}
        <div className="hidden sm:flex items-center gap-1">
          {['Features', 'About', 'Open Source'].map((item) => {
            const sectionId = item.toLowerCase().replace(' ', '');

            return (
              <a
                key={item}
                href={`#${sectionId}`}
                className={`
                  px-4 py-2 rounded-lg text-sm font-medium transition-all
                  ${
                    activeSection === sectionId
                      ? 'text-blue-500 border-b-2 border-blue-500'
                      : 'text-gray-700 dark:text-gray-400'
                  }
                  hover:bg-black/5 dark:hover:bg-white/5
                  hover:text-black dark:hover:text-white
                `}
              >
                {item}
              </a>
            );
          })}
        </div>

        <div className="w-px h-6 bg-black/10 dark:bg-white/10 mx-2 hidden sm:block" />

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          aria-label="Toggle dark/light mode"
          className="
            p-2 rounded-lg transition-colors
            text-gray-600 dark:text-gray-400
            hover:bg-black/10 dark:hover:bg-white/10
            hover:text-black dark:hover:text-white
          "
        >
          {isDark ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        {/* GitHub */}
        <a
          href="https://github.com/aayush-1709/Drive-Detect"
          target="_blank"
          rel="noopener noreferrer"
          className="
            p-2 rounded-lg transition-colors
            text-gray-600 dark:text-gray-400
            hover:bg-black/10 dark:hover:bg-white/10
            hover:text-black dark:hover:text-white
          "
        >
          <Github size={20} />
        </a>

        {/* CTA */}
        <Link
          to="/app"
          className="
            flex items-center gap-2 px-4 py-2
            bg-blue-600 hover:bg-blue-500
            text-white text-sm font-semibold rounded-xl
            shadow-lg shadow-blue-500/20
            transition-all active:scale-95
          "
        >
          <Terminal size={16} />
          <span className="hidden sm:block">Run Program</span>
        </Link>
      </motion.nav>
    </div>
  );
};
