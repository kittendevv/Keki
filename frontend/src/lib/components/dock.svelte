<script lang="ts">
	import { resolve } from '$app/paths';

	import MagnifyingGlassIcon from 'phosphor-svelte/lib/MagnifyingGlassIcon';
	import ClockIcon from 'phosphor-svelte/lib/ClockIcon';
	import GearIcon from 'phosphor-svelte/lib/GearIcon';
</script>

<div class="dock">
	<a href={resolve('/search')} class="dock-active">
		<MagnifyingGlassIcon weight="bold" />
		<span class="dock-label">Search</span>
	</a>

	<a href={resolve('/')}>
		<ClockIcon weight="bold" />
		<span class="dock-label">History</span>
	</a>

	<a href={resolve('/')}>
		<GearIcon />
		<span class="dock-label">Settings</span>
	</a>
</div>
