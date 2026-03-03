import { Navbar } from '../components/layout/Navbar';
import { Hero } from '../components/sections/Hero';
import { About } from '../components/sections/About';
import { Features } from '../components/sections/Features';
import { OpenSource } from '../components/sections/OpenSource';
import { Footer } from '../components/layout/Footer';
import { FAQ } from '../components/sections/FAQ';

export function Landing() {
    return (
        <div className="min-h-screen bg-white dark:bg-gray-950 font-sans selection:bg-blue-100 selection:text-blue-900">
            <Navbar />
            <main>
                <Hero />
                <About />
                <Features />
                <OpenSource />
                <FAQ />
            </main>
            <Footer />
        </div>
    );
}
