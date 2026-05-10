<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/stores';

	import HouseIcon from 'phosphor-svelte/lib/HouseIcon';
	import MagnifyingGlassIcon from 'phosphor-svelte/lib/MagnifyingGlassIcon';
	import ClockIcon from 'phosphor-svelte/lib/ClockIcon';
	import GearIcon from 'phosphor-svelte/lib/GearIcon';
</script>

<div class="dock dock-xl">
	<a href={resolve('/')} class:dock-active={$page.url.pathname === '/'}>
		<HouseIcon weight="duotone" size={18} />
		<span class="dock-label">Home</span>
	</a>

	<a href={resolve('/search')} class:dock-active={$page.url.pathname === '/search'}>
		<MagnifyingGlassIcon weight="duotone" size={18} />
		<span class="dock-label">Search</span>
	</a>

	<a href={resolve('/history')} class:dock-active={$page.url.pathname === '/history'}>
		<ClockIcon weight="duotone" size={18} />
		<span class="dock-label">History</span>
	</a>

	<a href={resolve('/settings')} class:dock-active={$page.url.pathname === '/settings'}>
		<GearIcon weight="duotone" size={18} />
		<span class="dock-label">Settings</span>
	</a>
</div>
