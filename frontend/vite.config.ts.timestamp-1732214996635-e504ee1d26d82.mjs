// vite.config.ts
import { defineConfig, loadEnv } from "file:///Users/sanid/Desktop/Interactivity/OpenHands/frontend/node_modules/vite/dist/node/index.js";
import viteTsconfigPaths from "file:///Users/sanid/Desktop/Interactivity/OpenHands/frontend/node_modules/vite-tsconfig-paths/dist/index.js";
import svgr from "file:///Users/sanid/Desktop/Interactivity/OpenHands/frontend/node_modules/vite-plugin-svgr/dist/index.js";
import { vitePlugin as remix } from "file:///Users/sanid/Desktop/Interactivity/OpenHands/frontend/node_modules/@remix-run/dev/dist/index.js";
import { configDefaults } from "file:///Users/sanid/Desktop/Interactivity/OpenHands/frontend/node_modules/vitest/dist/config.js";
var __vite_injected_original_dirname = "/Users/sanid/Desktop/Interactivity/OpenHands/frontend";
var vite_config_default = defineConfig(({ mode }) => {
  const {
    VITE_BACKEND_HOST = "127.0.0.1:3000",
    VITE_USE_TLS = "false",
    VITE_FRONTEND_PORT = "3001",
    VITE_INSECURE_SKIP_VERIFY = "false"
  } = loadEnv(mode, process.cwd());
  const USE_TLS = VITE_USE_TLS === "true";
  const INSECURE_SKIP_VERIFY = VITE_INSECURE_SKIP_VERIFY === "true";
  const PROTOCOL = USE_TLS ? "https" : "http";
  const WS_PROTOCOL = USE_TLS ? "wss" : "ws";
  const API_URL = `${PROTOCOL}://${VITE_BACKEND_HOST}/`;
  const WS_URL = `${WS_PROTOCOL}://${VITE_BACKEND_HOST}/`;
  const FE_PORT = Number.parseInt(VITE_FRONTEND_PORT, 10);
  const unpackClientDirectory = async () => {
    const fs = await import("fs");
    const path = await import("path");
    const buildDir = path.resolve(__vite_injected_original_dirname, "build");
    const clientDir = path.resolve(buildDir, "client");
    const files = await fs.promises.readdir(clientDir);
    await Promise.all(
      files.map(
        (file) => fs.promises.rename(
          path.resolve(clientDir, file),
          path.resolve(buildDir, file)
        )
      )
    );
    await fs.promises.rmdir(clientDir);
  };
  return {
    plugins: [
      !process.env.VITEST && remix({
        future: {
          v3_fetcherPersist: true,
          v3_relativeSplatPath: true,
          v3_throwAbortReason: true
        },
        appDirectory: "src",
        buildEnd: unpackClientDirectory,
        ssr: false
      }),
      viteTsconfigPaths(),
      svgr()
    ],
    server: {
      port: FE_PORT,
      proxy: {
        "/api": {
          target: API_URL,
          changeOrigin: true,
          secure: !INSECURE_SKIP_VERIFY
        },
        "/ws": {
          target: WS_URL,
          ws: true,
          changeOrigin: true,
          secure: !INSECURE_SKIP_VERIFY
        }
      }
    },
    ssr: {
      noExternal: ["react-syntax-highlighter"]
    },
    clearScreen: false,
    test: {
      environment: "jsdom",
      setupFiles: ["vitest.setup.ts"],
      exclude: [...configDefaults.exclude, "tests"],
      coverage: {
        reporter: ["text", "json", "html", "lcov", "text-summary"],
        reportsDirectory: "coverage",
        include: ["src/**/*.{ts,tsx}"]
      }
    }
  };
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvVXNlcnMvc2FuaWQvRGVza3RvcC9JbnRlcmFjdGl2aXR5L09wZW5IYW5kcy9mcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiL1VzZXJzL3NhbmlkL0Rlc2t0b3AvSW50ZXJhY3Rpdml0eS9PcGVuSGFuZHMvZnJvbnRlbmQvdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL1VzZXJzL3NhbmlkL0Rlc2t0b3AvSW50ZXJhY3Rpdml0eS9PcGVuSGFuZHMvZnJvbnRlbmQvdml0ZS5jb25maWcudHNcIjsvKiBlc2xpbnQtZGlzYWJsZSBpbXBvcnQvbm8tZXh0cmFuZW91cy1kZXBlbmRlbmNpZXMgKi9cbi8vLyA8cmVmZXJlbmNlIHR5cGVzPVwidml0ZXN0XCIgLz5cbi8vLyA8cmVmZXJlbmNlIHR5cGVzPVwidml0ZS1wbHVnaW4tc3Znci9jbGllbnRcIiAvPlxuaW1wb3J0IHsgZGVmaW5lQ29uZmlnLCBsb2FkRW52IH0gZnJvbSBcInZpdGVcIjtcbmltcG9ydCB2aXRlVHNjb25maWdQYXRocyBmcm9tIFwidml0ZS10c2NvbmZpZy1wYXRoc1wiO1xuaW1wb3J0IHN2Z3IgZnJvbSBcInZpdGUtcGx1Z2luLXN2Z3JcIjtcbmltcG9ydCB7IHZpdGVQbHVnaW4gYXMgcmVtaXggfSBmcm9tIFwiQHJlbWl4LXJ1bi9kZXZcIjtcbmltcG9ydCB7IGNvbmZpZ0RlZmF1bHRzIH0gZnJvbSBcInZpdGVzdC9jb25maWdcIjtcblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKCh7IG1vZGUgfSkgPT4ge1xuICBjb25zdCB7XG4gICAgVklURV9CQUNLRU5EX0hPU1QgPSBcIjEyNy4wLjAuMTozMDAwXCIsXG4gICAgVklURV9VU0VfVExTID0gXCJmYWxzZVwiLFxuICAgIFZJVEVfRlJPTlRFTkRfUE9SVCA9IFwiMzAwMVwiLFxuICAgIFZJVEVfSU5TRUNVUkVfU0tJUF9WRVJJRlkgPSBcImZhbHNlXCIsXG4gIH0gPSBsb2FkRW52KG1vZGUsIHByb2Nlc3MuY3dkKCkpO1xuXG4gIGNvbnN0IFVTRV9UTFMgPSBWSVRFX1VTRV9UTFMgPT09IFwidHJ1ZVwiO1xuICBjb25zdCBJTlNFQ1VSRV9TS0lQX1ZFUklGWSA9IFZJVEVfSU5TRUNVUkVfU0tJUF9WRVJJRlkgPT09IFwidHJ1ZVwiO1xuICBjb25zdCBQUk9UT0NPTCA9IFVTRV9UTFMgPyBcImh0dHBzXCIgOiBcImh0dHBcIjtcbiAgY29uc3QgV1NfUFJPVE9DT0wgPSBVU0VfVExTID8gXCJ3c3NcIiA6IFwid3NcIjtcblxuICBjb25zdCBBUElfVVJMID0gYCR7UFJPVE9DT0x9Oi8vJHtWSVRFX0JBQ0tFTkRfSE9TVH0vYDtcbiAgY29uc3QgV1NfVVJMID0gYCR7V1NfUFJPVE9DT0x9Oi8vJHtWSVRFX0JBQ0tFTkRfSE9TVH0vYDtcbiAgY29uc3QgRkVfUE9SVCA9IE51bWJlci5wYXJzZUludChWSVRFX0ZST05URU5EX1BPUlQsIDEwKTtcblxuICAvKipcbiAgICogVGhpcyBzY3JpcHQgaXMgdXNlZCB0byB1bnBhY2sgdGhlIGNsaWVudCBkaXJlY3RvcnkgZnJvbSB0aGUgZnJvbnRlbmQgYnVpbGQgZGlyZWN0b3J5LlxuICAgKiBSZW1peCBTUEEgbW9kZSBidWlsZHMgdGhlIGNsaWVudCBkaXJlY3RvcnkgaW50byB0aGUgYnVpbGQgZGlyZWN0b3J5LiBUaGlzIGZ1bmN0aW9uXG4gICAqIG1vdmVzIHRoZSBjb250ZW50cyBvZiB0aGUgY2xpZW50IGRpcmVjdG9yeSB0byB0aGUgYnVpbGQgZGlyZWN0b3J5IGFuZCB0aGVuIHJlbW92ZXMgdGhlXG4gICAqIGNsaWVudCBkaXJlY3RvcnkuXG4gICAqXG4gICAqIFRoaXMgc2NyaXB0IGlzIHVzZWQgaW4gdGhlIGJ1aWxkRW5kIGZ1bmN0aW9uIG9mIHRoZSBWaXRlIGNvbmZpZy5cbiAgICovXG4gIGNvbnN0IHVucGFja0NsaWVudERpcmVjdG9yeSA9IGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBmcyA9IGF3YWl0IGltcG9ydChcImZzXCIpO1xuICAgIGNvbnN0IHBhdGggPSBhd2FpdCBpbXBvcnQoXCJwYXRoXCIpO1xuXG4gICAgY29uc3QgYnVpbGREaXIgPSBwYXRoLnJlc29sdmUoX19kaXJuYW1lLCBcImJ1aWxkXCIpO1xuICAgIGNvbnN0IGNsaWVudERpciA9IHBhdGgucmVzb2x2ZShidWlsZERpciwgXCJjbGllbnRcIik7XG5cbiAgICBjb25zdCBmaWxlcyA9IGF3YWl0IGZzLnByb21pc2VzLnJlYWRkaXIoY2xpZW50RGlyKTtcbiAgICBhd2FpdCBQcm9taXNlLmFsbChcbiAgICAgIGZpbGVzLm1hcCgoZmlsZSkgPT5cbiAgICAgICAgZnMucHJvbWlzZXMucmVuYW1lKFxuICAgICAgICAgIHBhdGgucmVzb2x2ZShjbGllbnREaXIsIGZpbGUpLFxuICAgICAgICAgIHBhdGgucmVzb2x2ZShidWlsZERpciwgZmlsZSksXG4gICAgICAgICksXG4gICAgICApLFxuICAgICk7XG5cbiAgICBhd2FpdCBmcy5wcm9taXNlcy5ybWRpcihjbGllbnREaXIpO1xuICB9O1xuXG4gIHJldHVybiB7XG4gICAgcGx1Z2luczogW1xuICAgICAgIXByb2Nlc3MuZW52LlZJVEVTVCAmJlxuICAgICAgICByZW1peCh7XG4gICAgICAgICAgZnV0dXJlOiB7XG4gICAgICAgICAgICB2M19mZXRjaGVyUGVyc2lzdDogdHJ1ZSxcbiAgICAgICAgICAgIHYzX3JlbGF0aXZlU3BsYXRQYXRoOiB0cnVlLFxuICAgICAgICAgICAgdjNfdGhyb3dBYm9ydFJlYXNvbjogdHJ1ZSxcbiAgICAgICAgICB9LFxuICAgICAgICAgIGFwcERpcmVjdG9yeTogXCJzcmNcIixcbiAgICAgICAgICBidWlsZEVuZDogdW5wYWNrQ2xpZW50RGlyZWN0b3J5LFxuICAgICAgICAgIHNzcjogZmFsc2UsXG4gICAgICAgIH0pLFxuICAgICAgdml0ZVRzY29uZmlnUGF0aHMoKSxcbiAgICAgIHN2Z3IoKSxcbiAgICBdLFxuICAgIHNlcnZlcjoge1xuICAgICAgcG9ydDogRkVfUE9SVCxcbiAgICAgIHByb3h5OiB7XG4gICAgICAgIFwiL2FwaVwiOiB7XG4gICAgICAgICAgdGFyZ2V0OiBBUElfVVJMLFxuICAgICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcbiAgICAgICAgICBzZWN1cmU6ICFJTlNFQ1VSRV9TS0lQX1ZFUklGWSxcbiAgICAgICAgfSxcbiAgICAgICAgXCIvd3NcIjoge1xuICAgICAgICAgIHRhcmdldDogV1NfVVJMLFxuICAgICAgICAgIHdzOiB0cnVlLFxuICAgICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcbiAgICAgICAgICBzZWN1cmU6ICFJTlNFQ1VSRV9TS0lQX1ZFUklGWSxcbiAgICAgICAgfSxcbiAgICAgIH0sXG4gICAgfSxcbiAgICBzc3I6IHtcbiAgICAgIG5vRXh0ZXJuYWw6IFtcInJlYWN0LXN5bnRheC1oaWdobGlnaHRlclwiXSxcbiAgICB9LFxuICAgIGNsZWFyU2NyZWVuOiBmYWxzZSxcbiAgICB0ZXN0OiB7XG4gICAgICBlbnZpcm9ubWVudDogXCJqc2RvbVwiLFxuICAgICAgc2V0dXBGaWxlczogW1widml0ZXN0LnNldHVwLnRzXCJdLFxuICAgICAgZXhjbHVkZTogWy4uLmNvbmZpZ0RlZmF1bHRzLmV4Y2x1ZGUsIFwidGVzdHNcIl0sXG4gICAgICBjb3ZlcmFnZToge1xuICAgICAgICByZXBvcnRlcjogW1widGV4dFwiLCBcImpzb25cIiwgXCJodG1sXCIsIFwibGNvdlwiLCBcInRleHQtc3VtbWFyeVwiXSxcbiAgICAgICAgcmVwb3J0c0RpcmVjdG9yeTogXCJjb3ZlcmFnZVwiLFxuICAgICAgICBpbmNsdWRlOiBbXCJzcmMvKiovKi57dHMsdHN4fVwiXSxcbiAgICAgIH0sXG4gICAgfSxcbiAg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
