import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cố định workspace root về thư mục này. Tránh Next suy luận nhầm root ra
  // ~/ khi có nhiều lockfile trên máy.
  turbopack: {
    root: import.meta.dirname,
  },
};

export default nextConfig;
