// Vite turns an import ending in "?url" into the address of the file in the build.
declare module "*?url" {
  const url: string;
  export default url;
}
