import os
import asyncio
from typing import List

import discord
from discord import app_commands
from discord.ext import commands
from decouple import config, UndefinedValueError

from utils.log import setup_logger

os.makedirs('logs', exist_ok=True)
logger = setup_logger()

class DiscOllama(commands.Bot):
    def __init__(self):        
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)

    async def setup_hook(self):
        await self.load_cogs()
        await self.tree.sync()

    async def load_cogs(self):
        for filename in os.listdir('./cogs'):
            if filename.endswith('.py'):
                try:
                    await self.load_extension(f'cogs.{filename[:-3]}')
                    logger.info(f'Loaded cog: {filename[:-3]}')
                except Exception as e:
                    logger.error(f'Failed to load cog {filename[:-3]}: {str(e)}')

    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Serving {len(self.guilds)} guilds')

async def main():
    try:
        TOKEN = config('DISCORD_TOKEN')
    except UndefinedValueError:
        logger.error("DISCORD_TOKEN not found in environment variables or .env file")
        raise SystemExit(1)

    bot = DiscOllama()
    async with bot:
        await bot.start(TOKEN)

if __name__ == '__main__':
    asyncio.run(main())