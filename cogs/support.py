import discord
from discord import app_commands
from discord.ext import commands
from utils.log import setup_logger
from utils.rag import RAGChatPipeline

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

logger = setup_logger()

Base = declarative_base()

class SupportTicket(Base):
    __tablename__ = 'support_tickets'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, unique=True, index=True)
    user_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Support(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.rag = RAGChatPipeline()
        self.engine = create_engine('sqlite:///database.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.defer()
        latency = round(self.bot.latency * 1000)
        await interaction.followup.send(f"Pong! Latency: {latency}ms")
        logger.info(f"Ping command executed by {interaction.user.name} (ID: {interaction.user.id}). Latency: {latency}ms")

    @app_commands.command(name="support", description="Create a support ticket")
    async def create_ticket(self, interaction: discord.Interaction):
        try:
            thread = await interaction.channel.create_thread(
                name=f"Support: {interaction.user.name}",
                type=discord.ChannelType.private_thread,
                reason="Support ticket created",
            )

            welcome_embed = discord.Embed(
                title="Support Ticket Created",
                description="Welcome to your support ticket. Please describe your issue, and our AI assistant will help you.",
                color=discord.Color.blue()
            )
            welcome_embed.add_field(name="Ticket Owner", value=interaction.user.mention, inline=False)
            welcome_embed.add_field(name="Instructions", value="Describe your issue clearly, and the AI will respond to your messages.", inline=False)
            welcome_embed.set_footer(text="Our AI assistant will do its best to provide support.")

            await thread.send(embed=welcome_embed)
            await interaction.response.send_message(f"Support ticket created in {thread.mention}", ephemeral=True)

            with self.Session() as session:
                new_ticket = SupportTicket(thread_id=str(thread.id), user_id=str(interaction.user.id))
                session.add(new_ticket)
                session.commit()

            logger.info(f"Support ticket created by {interaction.user.name} (ID: {interaction.user.id}) in channel {thread.name} (ID: {thread.id})")

        except discord.errors.Forbidden:
            logger.error(f"Permission error creating support ticket for user {interaction.user.id}")
            await interaction.response.send_message("I don't have permission to create a support ticket. Please contact an administrator.", ephemeral=True)
        except discord.errors.HTTPException as e:
            logger.error(f"HTTP error creating support ticket: {str(e)}")
            await interaction.response.send_message("An error occurred while creating the support ticket. Please try again later.", ephemeral=True)
        except Exception as e:
            logger.error(f"Unexpected error in create_ticket command: {str(e)}")
            await interaction.response.send_message("An unexpected error occurred. Please contact an administrator.", ephemeral=True)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not isinstance(message.channel, discord.Thread) or message.author.bot:
            return

        thread_id = str(message.channel.id)

        with self.Session() as session:
            ticket = session.query(SupportTicket).filter_by(thread_id=thread_id).first()
            if not ticket or str(message.author.id) != ticket.user_id:
                return

        try:
            async with message.channel.typing():
                response = self.rag.chat(message.content, thread_id)
            await message.channel.send(response)
            logger.info(f"RAG response sent in ticket {thread_id} for user {message.author.id}")
        except Exception as e:
            logger.error(f"Error generating RAG response in ticket {thread_id}: {str(e)}")
            await message.channel.send("I apologize, but I encountered an error while processing your request. Please try again or contact a human administrator if the issue persists.")

    def cog_unload(self):
        self.rag.close()
        self.engine.dispose()

async def setup(bot: commands.Bot):
    await bot.add_cog(Support(bot))
    logger.info("Support cog loaded successfully")