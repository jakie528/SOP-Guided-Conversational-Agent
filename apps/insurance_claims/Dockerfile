FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0
COPY --chown=node:node package.json ./
COPY --chown=node:node src ./src
COPY --chown=node:node public ./public
COPY --chown=node:node fixtures ./fixtures
USER node
EXPOSE 3000
CMD ["node", "src/server.js"]
