import { defineConfig } from "vite";

// GitHub Pagesのプロジェクトページはリポジトリ名のサブパス配下で配信されるため、
// デプロイ時はVITE_BASE環境変数でベースパスを上書きする(ローカル開発では未設定のため"/"のまま)
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
});
