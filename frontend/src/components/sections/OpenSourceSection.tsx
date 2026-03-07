import React from 'react';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { Github, Star, GitBranch, Terminal } from 'lucide-react';

export const OpenSourceSection = () => {
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

if (!res.ok) {
  throw new Error("GitHub API request failed");
}

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

        {/* Contribution Guide */}
        <div className="mt-20 max-w-3xl mx-auto text-center">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Contribute to DriveDetect
          </h3>

          <p className="text-gray-600 dark:text-gray-400 mb-6">
            DriveDetect is an open-source project welcoming developers,
            researchers, and students. You can contribute by improving the UI,
            optimizing the model, fixing bugs, or adding new features.
          </p>

          <a
            href="https://github.com/aayush-1709/Drive-Detect"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-6 py-3 rounded-xl border border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white transition"
          >
            View Repository
          </a>
        </div>


        {/* Contribution Steps */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">

          <div className="p-6 rounded-xl border border-black/10 dark:border-white/10 bg-black/5 dark:bg-[#0a0a0a]">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
              1. Fork Repository
            </h4>
            <p className="text-sm text-gray-500">
              Fork the repository to create your own copy and start experimenting.
            </p>
          </div>

          <div className="p-6 rounded-xl border border-black/10 dark:border-white/10 bg-black/5 dark:bg-[#0a0a0a]">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
              2. Create Feature Branch
            </h4>
            <p className="text-sm text-gray-500">
              Create a new branch for your improvement or bug fix.
            </p>
          </div>

          <div className="p-6 rounded-xl border border-black/10 dark:border-white/10 bg-black/5 dark:bg-[#0a0a0a]">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
              3. Submit Pull Request
            </h4>
            <p className="text-sm text-gray-500">
              Open a pull request describing your changes and improvements.
            </p>
          </div>

        </div>


        {/* Beginner Issues */}
        <div className="mt-20 text-center max-w-3xl mx-auto">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Start with Beginner Issues
          </h3>

          <p className="text-gray-600 dark:text-gray-400 mb-6">
            If you're new to the project, explore open issues labeled for
            beginners and start contributing right away.
          </p>

          <a
            href="https://github.com/aayush-1709/Drive-Detect/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-6 py-3 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition"
          >
            Browse Open Issues
          </a>
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
