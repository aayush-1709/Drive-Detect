import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { Github, Star, GitBranch, Terminal } from 'lucide-react';

export const OpenSource = () => {
  const [repoStats, setRepoStats] = useState({
  stars: "0",
  forks: "0",
  license: "Loading..."
});

const [loading, setLoading] = useState(true);
useEffect(() => {
  const fetchRepoStats = async () => {
    try {
      const res = await fetch(
        "https://api.github.com/repos/aayush-1709/Drive-Detect"
      );
      const data = await res.json();

      setRepoStats({
  stars: data.stargazers_count?.toString() || "0",
  forks: data.forks_count?.toString() || "0",
  license:
    data.license?.spdx_id && data.license.spdx_id !== "NOASSERTION"
      ? data.license.spdx_id
      : "MIT"
});

      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch GitHub stats:", error);
      setLoading(false);
    }
  };

  fetchRepoStats();
}, []);
  return (
    <section
      id="opensource"
      className="py-24 bg-transparent text-center relative overflow-hidden border-t border-black/10 dark:border-white/5"
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(59,130,246,0.05)_0%,transparent_70%)] pointer-events-none"></div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="inline-flex items-center justify-center p-4 mb-8 bg-black/5 dark:bg-white/5 rounded-2xl border border-black/10 dark:border-white/10 shadow-lg">
            <Github size={40} className="text-gray-900 dark:text-white" />
          </div>

          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
            Open Code. <span className="text-gray-500 dark:text-gray-400">Open Roads.</span>
          </h2>

          <p className="text-xl text-gray-600 dark:text-gray-400 mb-10 max-w-2xl mx-auto font-light">
            We believe safety systems should be transparent. Inspect our models, contribute to the dataset,
            or fork the engine for your own robotics projects.
          </p>

          <a
            href="https://github.com/aayush-1709/Drive-Detect"
            target="_blank"
            rel="noopener noreferrer"
            className="
              inline-flex items-center gap-3 px-8 py-4 rounded-full font-bold transition-all
              bg-black text-white hover:bg-gray-800
              dark:bg-white dark:text-black dark:hover:bg-gray-200
              shadow-[0_0_20px_rgba(0,0,0,0.15)] dark:shadow-[0_0_20px_rgba(255,255,255,0.3)]
            "
          >
            <Github size={20} />
            <span>GITHUB REPOSITORY</span>
          </a>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
  <StatCard
    icon={<Star size={24} />}
    value={loading ? "..." : repoStats.stars}
    label="Stars"
  />

  <StatCard
    icon={<GitBranch size={24} />}
    value={loading ? "..." : repoStats.forks}
    label="Forks"
  />

  <StatCard
    icon={<Terminal size={24} />}
    value={repoStats.license}
    label="License"
  />
</div>
      </div>
    </section>
  );
};

const StatCard = ({
  icon,
  value,
  label,
}: {
  icon: React.ReactNode;
  value: string | number;
  label: string;
}) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="
        p-8 rounded-2xl transition-colors group
        bg-black/5 dark:bg-[#0a0a0a]
        border border-black/10 dark:border-white/5
        hover:border-black/20 dark:hover:border-white/20
      "
    >
      <div className="mb-4 text-gray-500 group-hover:text-blue-500 transition-colors flex justify-center">
        {icon}
      </div>

      <div className="text-4xl font-bold text-gray-900 dark:text-white mb-2 tracking-tight">
        {value}
      </div>

      <div className="text-xs text-gray-500 dark:text-gray-400 font-mono uppercase tracking-widest">
        {label}
      </div>
    </motion.div>
  );
};
